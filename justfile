set dotenv-load
set default-list

dataset_dir := env("DATASET_DIR")

data:
    rm -rf {{ dataset_dir }}
    mkdir -p {{ dataset_dir }}
    echo '*' > {{ dataset_dir }}/.gitignore
    ( cd {{ dataset_dir }} \
        && wcurl \
        https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz \
        https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz \
        https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz \
        https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz \
        && gunzip *.gz \
    )

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
