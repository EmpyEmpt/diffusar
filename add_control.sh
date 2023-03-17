SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/src/models/tool_add_control_sd21.py $SCRIPT_DIR/models/v2-1-pruned-ema.ckpt $SCRIPT_DIR/models/control_v2-1_ini.ckpt