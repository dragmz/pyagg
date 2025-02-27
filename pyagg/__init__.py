import csv
import random
import struct
import sys
import argparse
import logging
import time
import typing
import algosdk

import numpy as np
import pyopencl as cl
import dataclasses
import json

from typing import Callable, List, NamedTuple, Optional
import os

KEY_SIZE = 32
BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


@dataclasses.dataclass
class Config:
    batch: int


class Batch(NamedTuple):
    start: float
    seeds: bytes
    run_kernel_event: cl.event_info


class Buffers:
    seed: cl.Buffer
    size: cl.Buffer
    prefix_length: cl.Buffer
    prefix: cl.Buffer
    counts: cl.Buffer


def default_batch_size(device: cl.Device, preferred_multiple: int):
    return (
        int(device.max_mem_alloc_size / 1024 / preferred_multiple) * preferred_multiple
    )


class Initializer:
    def __init__(
        self,
        ctx: cl.Context,
        device: cl.Device,
        source: str,
        encoded: List[bytes],
        prefix_bytes: bytes,
        prefix_length_bytes: bytes,
        batch_size: Optional[int] = None,
        silent: bool = False,
    ):
        self.device = device
        self.ctx = ctx

        self.source = source
        self.encoded = encoded
        self.prefix_bytes = prefix_bytes
        self.prefix_length_bytes = prefix_length_bytes
        self.batch_size = batch_size
        self.silent = silent

        self.kernel_init = KernelInitializer(ctx, source)

    def __enter__(self):
        def _print(msg: str, *args):
            if not self.silent:
                print(msg, *args)

        _print("Preparing kernel. This may take several minutes on first run..")
        self.kernel = self.kernel_init.__enter__()

        batch_size = self.batch_size
        preferred_multiple = self.kernel.preferred_work_group_size_multiple(self.device)

        if not batch_size:
            batch_size = default_batch_size(self.device, preferred_multiple)
            _print("Batch size not specified. Using default value: %d" % batch_size)
        else:
            _print("Using batch size: %d" % batch_size)

        if batch_size % preferred_multiple != 0:
            raise ValueError("Batch size must be a multiple of %d" % preferred_multiple)

        count = len(self.encoded)
        count_bytes = struct.pack("<H" if self.device.endian_little else ">H", count)

        buffers = Buffers()

        mf = cl.mem_flags
        buffers.seed = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=batch_size * KEY_SIZE
        )
        buffers.size = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=len(count_bytes)
        )
        buffers.prefix_length = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=len(self.encoded)
        )
        buffers.prefix = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=len(self.prefix_bytes)
        )
        buffers.counts = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=batch_size)

        queue: cl.CommandQueue = cl.CommandQueue(self.ctx)

        cl.enqueue_copy(queue, buffers.size, count_bytes)
        cl.enqueue_copy(queue, buffers.prefix_length, self.prefix_length_bytes)
        cl.enqueue_copy(queue, buffers.prefix, self.prefix_bytes)

        self.queue = queue
        self.buffers = buffers

        self.g = Generator(
            batch_size=batch_size,
            count=count,
            device=self.device,
            kernel=self.kernel_init.kernel.kernel,
            queue=queue,
            buffers=buffers,
        )

        return self.g

    def __exit__(self, exc_type, exc_value, traceback):
        self.queue.finish()
        del self.queue

        self.buffers.prefix_length.release()
        self.buffers.prefix.release()
        self.buffers.seed.release()
        self.buffers.counts.release()

        self.kernel_init.__exit__(exc_type, exc_value, traceback)


class Result:
    total: int
    found: int
    duration: float
    batch_size: int
    batch_duration: float


class Callbacks:
    on_batch: Callable


class Kernel:
    def __init__(self, program: cl.Program, kernel: cl.Kernel):
        self.program = program
        self.kernel = kernel

    def preferred_work_group_size_multiple(self, device: cl.Device):
        return self.kernel.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device=device
        )


class KernelInitializer:
    def __init__(self, ctx: cl.Context, source: str):
        self.ctx = ctx
        self.source = source

    def __enter__(self) -> Kernel:
        program = build_program_from_source(self.ctx, self.source)
        kernel = program.ed25519_create_keypair

        self.kernel = Kernel(program=program, kernel=kernel)

        return self.kernel

    def __exit__(self, exc_type, exc_value, traceback):
        del self.kernel.kernel
        del self.kernel.program


class Generator:
    def __init__(
        self,
        batch_size: int,
        count: int,
        device: cl.Device,
        kernel: cl.Kernel,
        queue: cl.CommandQueue,
        buffers: Buffers,
    ):
        self.batch_size = batch_size
        self.count = count
        self.device = device
        self.kernel = kernel
        self.queue = queue
        self.buffers = buffers

    def __send_next_batch(self) -> Batch:
        seeds = os.urandom(self.batch_size * KEY_SIZE)
        cl.enqueue_copy(self.queue, self.buffers.seed, seeds, is_blocking=False)
        batch = Batch(
            start=time.perf_counter(),
            seeds=seeds,
            run_kernel_event=self.kernel(
                self.queue,
                [self.batch_size],
                None,
                self.buffers.seed,
                self.buffers.size,
                self.buffers.prefix_length,
                self.buffers.prefix,
                self.buffers.counts,
            ),
        )

        return batch

    def generate(
        self,
        count: int = 1,
        on_found: Optional[typing.Callable[[bytes], None]] = None,
        on_update: Optional[typing.Callable[[Result], None]] = None,
    ) -> Result:
        batch = self.__send_next_batch()
        start: float = time.perf_counter()

        total = 0
        found = 0

        left = count
        counts = np.zeros(self.batch_size, dtype="uint8")

        result = Result()
        result.batch_size = self.batch_size

        def update_result(result_batch: Batch):
            result.total = total
            result.found = found

            now = time.perf_counter()
            result.duration = now - start
            result.batch_duration = now - result_batch.start

        while found < left:
            copy_counts = cl.enqueue_copy(
                self.queue, counts, self.buffers.counts, is_blocking=False
            )

            next_batch = self.__send_next_batch()

            cl.wait_for_events([batch.run_kernel_event, copy_counts])

            total += self.batch_size

            seeds = batch.seeds

            if np.any(counts):
                indices = np.nonzero(counts)[0]
                for i in indices:
                    index = i // self.count
                    found += 1

                    if on_found:
                        on_found(
                            bytes(seeds[index * KEY_SIZE : (index + 1) * KEY_SIZE])
                        )

                    if found == left:
                        break

            update_result(batch)

            batch = next_batch

            if on_update:
                try:
                    on_update(result)
                except Exception:
                    break

        update_result(batch)

        cl.wait_for_events([batch.run_kernel_event])

        return result


class Context:
    def __init__(self, device: cl.Device = None):
        if device is None:
            platforms: typing.List[cl.Platform] = cl.get_platforms()

            for platform in platforms:
                if platform.name == "NVIDIA CUDA":
                    devices = platform.get_devices()
                    if len(devices) > 0:
                        device = devices[0]
                        break

        if device is None:
            ctx: cl.Context = cl.create_some_context()
            device = ctx.devices[0]
        else:
            ctx = cl.Context(devices=[device])
            device = device

        self.ctx = ctx
        self.device = device

    @property
    def device_name(self):
        return self.device.name

    def prepare(
        self,
        source: str,
        prefixes: List[str],
        batch_size: Optional[int] = None,
        silent: bool = False,
    ) -> Initializer:
        if not prefixes:
            raise ValueError("No prefixes specified")

        for p in prefixes:
            validate_prefix(p)

        encoded = [p.encode("utf-8") for p in prefixes]
        prefix_length_bytes = bytes([len(e) for e in encoded])

        encoded_to_64byte_chunks = [e + b"\0" * (64 - len(e)) for e in encoded]
        prefix_bytes = b"".join(encoded_to_64byte_chunks)

        g = Initializer(
            ctx=self.ctx,
            device=self.device,
            source=source,
            encoded=encoded,
            prefix_bytes=prefix_bytes,
            prefix_length_bytes=prefix_length_bytes,
            batch_size=batch_size,
            silent=silent,
        )

        return g


def mnemonic_to_json(mnemonic: str):
    @dataclasses.dataclass
    class MnemonicQR:
        version: str
        mnemonic: str

    m = MnemonicQR(version="1.0", mnemonic=mnemonic)
    return json.dumps(dataclasses.asdict(m))


def read_kernel_source_from_path(kernel_path: str) -> str:
    if kernel_path:
        with open(kernel_path, mode="r") as f:
            return f.read()
    else:
        import importlib.resources

        kernel_file = importlib.resources.files("pyagg") / "kernel.cl"
        with kernel_file.open(mode="r") as f:
            return f.read()


def build_program_from_source(ctx: cl.Context, source: str):
    return cl.Program(ctx, source).build(options="-cl-std=CL3.0")


def validate_prefix(p: str) -> None:
    if len(p) > 54:
        raise ValueError(
            "Prefix '%s' is too long. The maximum length is 54 characters" % p
        )

    for i in range(len(p)):
        c = p[i]
        if c not in BASE32_ALPHABET:
            raise ValueError("Prefix '%s' contains invalid character: '%s'" % (p, c))


def decode_key(key: bytes) -> typing.Tuple[str, str, str]:
    phrases = algosdk.mnemonic._from_key(key)
    pk = algosdk.mnemonic.to_private_key(phrases)
    addr = algosdk.account.address_from_private_key(pk)

    return phrases, pk, addr


def get_pyagg_config_path():
    return os.path.join(os.path.expanduser("~"), ".pyagg", "config.json")


def save_pyagg_config(config: Config):
    path = get_pyagg_config_path()
    print("Saving configuration to %s" % path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w+") as f:
        json.dump(dataclasses.asdict(config), f)


def try_load_pyagg_config() -> typing.Optional[Config]:
    try:
        with open(get_pyagg_config_path(), "r") as f:
            return Config(**json.load(f))
    except FileNotFoundError:
        return None


def read_prefixes(prefix: str, file: str) -> List[str]:
    prefixes: List[str] = []

    if prefix:
        for p in prefix:
            items = p[0].split(",")
            prefixes.extend(items)

    if file:
        with open(file, mode="r") as f:
            lines = f.readlines()
            prefixes.extend([line.strip() for line in lines])

    prefixes = [p.upper() for p in prefixes]

    return prefixes


def pyagg_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str)
    parser.add_argument("--prefix", type=str, nargs="+", action="append")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.batch:
        config = try_load_pyagg_config()
        if config is not None:
            args.batch = config.batch

    if args.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

    prefixes = read_prefixes(args.prefix, args.file)

    if len(prefixes) == 0:
        print("No prefixes specified")
        parser.print_help()
        exit(1)

    if args.count < 1:
        print("Count must be greater than 0")
        parser.print_help()
        exit(1)

    ctx = Context()

    batch_size = args.batch

    source = read_kernel_source_from_path(args.kernel)

    with ctx.prepare(source=source, prefixes=prefixes, batch_size=batch_size) as g:
        print(
            "Looking for %d %s with %s: %s"
            % (
                args.count,
                "key" if args.count == 1 else "keys",
                "prefix" if len(prefixes) == 1 else "prefixes",
                ", ".join(prefixes),
            )
        )

        def on_result(result: Result):
            print(
                "Found %d keys, time: %.02fs, total: %d keys, avg: %d keys/s, last batch: %d keys/s"
                % (
                    result.found,
                    result.duration,
                    result.total,
                    result.total / result.duration,
                    result.batch_size / result.batch_duration,
                )
            )

        update_cb = on_result if args.benchmark else None

        def on_found(key: bytes):
            (phrases, pk, addr) = decode_key(key)
            print("%s,%s" % (addr, phrases))

        current_perf = g.generate(
            count=args.count, on_update=update_cb, on_found=on_found
        )

        avg = 0
        if current_perf.duration > 0:
            avg = current_perf.total / current_perf.duration

        print("--- Benchmark Result")
        print("Device: %s" % ctx.device_name)
        print(
            "Total: %d keys, matching: %d, time: %.02fs, avg: %d keys/s"
            % (current_perf.total, current_perf.found, current_perf.duration, avg)
        )


def pyagg_optimize_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--file", type=str)
    parser.add_argument("--prefix", type=str, nargs="+", action="append")

    args = parser.parse_args()

    prefixes = read_prefixes(args.prefix, args.file)
    if not prefixes:
        prefixes = ["AAAAAAAA"]

    ctx = Context()

    if args.output:

        def file_output(batch_size: int, result: float):
            csv_writer.writerow([batch_size, result])

        f = open(args.output, "w+", newline="", encoding="utf-8")
        csv_writer = csv.writer(f)
        output = file_output
    else:

        def dummy_output(batch_size: int, result: float):
            pass

        output = dummy_output

    source = read_kernel_source_from_path(args.kernel)

    with KernelInitializer(ctx.ctx, source) as k:
        preferred_work_group_size_multiple = k.preferred_work_group_size_multiple(
            ctx.device
        )

    def step(step_batch_size: int):
        def on_update(update: Result):
            if update.total > update.total / update.duration:
                raise Exception("Done")

        with ctx.prepare(
            source=source, prefixes=prefixes, batch_size=step_batch_size, silent=True
        ) as g:
            result = g.generate(count=sys.maxsize, on_update=on_update)

            performance = result.total / result.duration
            output(step_batch_size, performance)

            return performance

    max_perf = 0
    max_perf_batch_size = preferred_work_group_size_multiple

    min_index = int(args.min / preferred_work_group_size_multiple)
    if min_index < 1:
        min_index = 1

    if args.max is None:
        max_index = int(
            default_batch_size(ctx.device, preferred_work_group_size_multiple)
            / preferred_work_group_size_multiple
        )
    else:
        max_index = int(args.max / preferred_work_group_size_multiple)

    print("Optimizing batch size for device: %s" % ctx.device_name)
    print(
        "Batch size range: %d - %d"
        % (
            min_index * preferred_work_group_size_multiple,
            max_index * preferred_work_group_size_multiple,
        )
    )
    print("Prefixes: %s" % ", ".join(prefixes))

    try:
        while True:
            index = random.randint(min_index, max_index)
            current_batch_size = index * preferred_work_group_size_multiple
            current_perf = step(current_batch_size)

            if current_perf > max_perf:
                max_perf = current_perf
                max_perf_batch_size = current_batch_size
                print(
                    "Max performance: %d keys/s, batch size: %d"
                    % (max_perf, max_perf_batch_size)
                )
    except KeyboardInterrupt:
        pass

    # ask if save
    if input("Save configuration (y/N): ") == "y":
        save_pyagg_config(Config(batch=max_perf_batch_size))
        print("Configuration saved")
