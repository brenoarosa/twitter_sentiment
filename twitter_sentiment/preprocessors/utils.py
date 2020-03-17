import lzma
import json
from typing import Generator

def read_jsonlines_lzma(filepath: str) -> Generator[dict, None, None]:
    try:
        with lzma.LZMAFile(filepath, 'r') as fin:
            for line in fin:
                try:
                    data = json.loads(line.decode("utf-8"))
                    yield data
                except ValueError:
                    pass
    except EOFError as exc:
        print("EOF Error: {}\n{}".format(filepath, exc))
    except lzma.LZMAError as exc:
        print("LZMA Error: {}\n{}".format(filepath, exc))
