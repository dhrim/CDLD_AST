from pathlib import Path
import os, json
PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get('CDLD_OUTPUT_DIR', str(Path.home()/'CDLD_AST_outputs'))).expanduser().resolve()
assert OUTPUT != PACKAGE and PACKAGE not in OUTPUT.parents, 'Keep outputs outside the repository.'
B = OUTPUT/'run'
W = OUTPUT/'review'
W.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((PACKAGE/'config/protocol.json').read_text())
