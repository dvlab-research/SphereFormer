# Modified from https://github.com/IrohXu/waymo_to_semanticKITTI

from lib.waymo2semantickitti import Waymo2SemanticKITTI
from lib.utils import parse_args

args = parse_args()

waymo_load_dir = args.load_dir
waymo_save_dir = args.save_dir

coverter = Waymo2SemanticKITTI(waymo_load_dir, waymo_save_dir)

coverter.convert_all_mp()
