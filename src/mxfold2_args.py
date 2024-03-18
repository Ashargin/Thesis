from pathlib import Path
import mxfold2

default_conf = Path(mxfold2.__file__).parents[0] / "models/TrainSetAB.conf"


class Mxfold2Args:
    seed = 0
    gpu = 0
    param = ""
    use_constraint = False
    result = None
    bpseq = None
    bpp = None
    model = "MixC"
    max_helix_length = 30
    embed_size = 0
    num_filters = [96]
    filter_size = [5]
    pool_size = [1]
    dilation = 0
    num_lstm_layers = 0
    num_lstm_units = 0
    num_transformer_layers = 0
    num_transformer_hidden_units = 2048
    num_transformer_att = 8
    num_paired_filters = [96]
    paired_filter_size = [5]
    num_hidden_units = [32]
    dropout_rate = 0.0
    fc_dropout_rate = 0.0
    num_att = 0
    pair_join = "cat"
    no_split_lr = False

    def __init__(self, conf=default_conf):
        txt = open(conf, "r").read()
        groups = txt.strip().split("--")
        conf_args = [g.strip().split("\n") for g in groups if g]
        names = [x[0] for x in conf_args]
        if "num-filters" in names:
            self.num_filters = []
        if "filter-size" in names:
            self.filter_size = []
        if "pool-size" in names:
            self.pool_size = []
        if "num-paired-filters" in names:
            self.num_paired_filters = []
        if "paired-filter-size" in names:
            self.paired_filter_size = []
        if "num-hidden-units" in names:
            self.num_hidden_units = []

        for name, val in conf_args:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val == "None":
                        val = None
                    elif val == "False":
                        val = False
                    elif val == "True":
                        val = True

            if name == "seed":
                self.seed = val
            elif name == "gpu":
                self.gpu = val
            elif name == "param":
                self.param = val
            elif name == "use-constraint":
                self.use_constraint = val
            elif name == "result":
                self.result = val
            elif name == "bpseq":
                self.bpseq = val
            elif name == "bpp":
                self.bpp = val
            elif name == "model":
                self.model = val
            elif name == "max-helix-length":
                self.max_helix_length = val
            elif name == "embed-size":
                self.embed_size = val
            elif name == "num-filters":
                self.num_filters.append(val)
            elif name == "filter-size":
                self.filter_size.append(val)
            elif name == "pool-size":
                self.pool_size.append(val)
            elif name == "dilation":
                self.dilation = val
            elif name == "num-lstm-layers":
                self.num_lstm_layers = val
            elif name == "num-lstm-units":
                self.num_lstm_units = val
            elif name == "num-transformer-layers":
                self.num_transformer_layers = val
            elif name == "num-transformer-hidden-units":
                self.num_transformer_hidden_units = val
            elif name == "num-transformer-att":
                self.num_transformer_att = val
            elif name == "num-paired-filters":
                self.num_paired_filters.append(val)
            elif name == "paired-filter-size":
                self.paired_filter_size.append(val)
            elif name == "num-hidden-units":
                self.num_hidden_units.append(val)
            elif name == "dropout-rate":
                self.dropout_rate = val
            elif name == "fc-dropout-rate":
                self.fc_dropout_rate = val
            elif name == "num-att":
                self.num_att = val
            elif name == "pair-join":
                self.pair_join = val
            elif name == "no-split-lr":
                self.no_split_lr = val
