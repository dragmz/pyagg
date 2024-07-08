## pyagg

Python Algorand GPU vanity address generator

### Installation

#### Install from the repository

````commandline
pip install git+https://github.com/dragmz/pyagg
````

### Usage

#### Optimize batch size

````commandline
pyagg-optimize
````

(hit ctrl+c to stop)

````
Optimizing batch size for device: NVIDIA GeForce GTX 950
Batch size range: 32 - 524224
Prefixes: AAAAAAAA
Max performance: 448615 keys/s, batch size: 157408
Max performance: 463908 keys/s, batch size: 38112
Max performance: 466800 keys/s, batch size: 351328
Save configuration (y/N):
Saving configuration to C:\Users\dragmz\.pyagg\config.json
Configuration saved
````

#### Run the generator

````commandline
pyagg --prefix TEST --count 3 --benchmark
````

````
(Address 1),(Mnemonic 1)
(Address 2),(Mnemonic 2)
(Address 3),(Mnemonic 3)
Total: 2097152, matching: 3, time: 7.384147644042969, avg: 284007 keys/s
````

#### Lookup multiple prefixes at once

````commandline
pyagg --prefix TEST,TEST2,TEST3 --count 10
````

#### Lookup prefixes from a file

````commandline
pyagg --file prefixes.txt --count 10
````

### Credits

This work has been performed with support from the Algorand Foundation xGov Grants Program.