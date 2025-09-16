optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "s4.src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "s4.src.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "s4.src.callbacks.timer.Timer",
    "params":                "s4.src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "s4.src.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

model = {
    # Backbones from this repo
    "model":                 "s4.src.models.sequence.backbones.model.SequenceModel",
    "unet":                  "s4.src.models.sequence.backbones.unet.SequenceUNet",
    "sashimi":               "s4.src.models.sequence.backbones.sashimi.Sashimi",
    "sashimi_standalone":    "s4.models.sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm":                  "s4.src.models.baselines.lstm.TorchLSTM",
    "gru":                   "s4.src.models.baselines.gru.TorchGRU",
    "unicornn":              "s4.src.models.baselines.unicornn.UnICORNN",
    "odelstm":               "s4.src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn":          "s4.src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn":            "s4.src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline":   "s4.src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn":             "s4.src.models.baselines.samplernn.SampleRNN",
    "dcgru":                 "s4.src.models.baselines.dcgru.DCRNNModel_classification",
    "dcgru_ss":              "s4.src.models.baselines.dcgru.DCRNNModel_nextTimePred",
    # Baseline CNNs
    "ckconv":                "s4.src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan":               "s4.src.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "denseinception":        "s4.src.models.baselines.dense_inception.DenseInception",
    "wavenet":               "s4.src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d":        "s4.src.models.baselines.resnet.TorchVisionResnet",  # 2D ResNet
    # Nonaka 1D CNN baselines
    "nonaka/resnet18":       "s4.src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception":      "s4.src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50":      "s4.src.models.baselines.nonaka.xresnet.xresnet1d50",
    # ViT Variants (note: small variant is taken from Tri, differs from original)
    "vit":                   "s4.models.baselines.vit.ViT",
    "vit_s_16":              "s4.src.models.baselines.vit_all.vit_small_patch16_224",
    "vit_b_16":              "s4.src.models.baselines.vit_all.vit_base_patch16_224",
    # Timm models
    "timm/convnext_base":    "s4.src.models.baselines.convnext_timm.convnext_base",
    "timm/convnext_small":   "s4.src.models.baselines.convnext_timm.convnext_small",
    "timm/convnext_tiny":    "s4.src.models.baselines.convnext_timm.convnext_tiny",
    "timm/convnext_micro":   "s4.src.models.baselines.convnext_timm.convnext_micro",
    "timm/resnet50":         "s4.src.models.baselines.resnet_timm.resnet50", # Can also register many other variants in resnet_timm
    "timm/convnext_tiny_3d": "s4.src.models.baselines.convnext_timm.convnext3d_tiny",
    # Segmentation models
    "convnext_unet_tiny":    "s4.src.models.segmentation.convnext_unet.convnext_tiny_unet",
}

layer = {
    "id":         "s4.src.models.sequence.base.SequenceIdentity",
    "lstm":       "s4.src.models.baselines.lstm.TorchLSTM",
    "standalone": "s4.models.s4.s4.S4Block",
    "s4d":        "s4.models.s4.s4d.S4D",
    "ffn":        "s4.src.models.sequence.modules.ffn.FFN",
    "sru":        "s4.src.models.sequence.rnns.sru.SRURNN",
    "rnn":        "s4.src.models.sequence.rnns.rnn.RNN",  # General RNN wrapper
    "conv1d":     "s4.src.models.sequence.convs.conv1d.Conv1d",
    "conv2d":     "s4.src.models.sequence.convs.conv2d.Conv2d",
    "mha":        "s4.src.models.sequence.attention.mha.MultiheadAttention",
    "vit":        "s4.src.models.sequence.attention.mha.VitAttention",
    "performer":  "s4.src.models.sequence.attention.linear.Performer",
    "lssl":       "s4.src.models.sequence.modules.lssl.LSSL",
    "s4":         "s4.src.models.sequence.modules.s4block.S4Block",
    "fftconv":    "s4.src.models.sequence.kernels.fftconv.FFTConv",
    "s4nd":       "s4.src.models.sequence.modules.s4nd.S4ND",
    "mega":       "s4.src.models.sequence.modules.mega.MegaBlock",
    "h3":         "s4.src.models.sequence.experimental.h3.H3",
    "h4":         "s4.src.models.sequence.experimental.h4.H4",
    # "packedrnn": "s4.models.sequence.rnns.packedrnn.PackedRNN",
}

layer_decay = {
    "convnext_timm_tiny": "s4.src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny",
}

model_state_hook = {
    "convnext_timm_tiny_2d_to_3d": "s4.src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d",
    "convnext_timm_tiny_s4nd_2d_to_3d": "s4.src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d",
}
