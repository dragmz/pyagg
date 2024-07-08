import pyagg


def test_generator():
    TEST_PREFIX = "A"

    ctx = pyagg.Context()
    with ctx.prepare(source=pyagg.read_kernel_source_from_path("kernel.cl"), prefixes=[TEST_PREFIX], silent=True) as g:
        count = 0

        def on_found(key: bytes):
            (phrases, pk, addr) = pyagg.decode_key(key)

            assert len(phrases.split(" ")) == 25
            assert len(addr) == 58

            assert addr.startswith(TEST_PREFIX)

            nonlocal count
            count += 1

        result = g.generate(on_found=on_found)

        assert result.found == 1
        assert count == 1
