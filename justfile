set dotenv-load

[private]
default:
    @just --list

check:
    prek run --all-files --no-group slow

test:
    prek run test --all-files --verbose

build *args:
    zig build -Doptimize=ReleaseFast {{ args }}

ci:
    prek run --all-files

run *args:
    zig build run {{ args }}
