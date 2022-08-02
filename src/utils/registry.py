optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "src.models.sequence.SequenceModel",
    "unet": "src.models.sequence.SequenceUNet",
    "sashimi": "src.models.sequence.sashimi.Sashimi",
    # Baseline RNNs
    "lstm": "src.models.baselines.lstm.TorchLSTM",
    "gru": "src.models.baselines.gru.TorchGRU",
    "unicornn": "src.models.baselines.unicornn.UnICORNN",
    "odelstm": "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn": "src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline": "src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn": "src.models.baselines.samplernn.SampleRNN",
    # Baseline CNNs
    "ckconv": "src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan": "src.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "wavenet": "src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d": "src.models.baselines.resnet.TorchVisionResnet",
    # Nonaka 1D CNN baselines
    "nonaka/resnet18": "src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception": "src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50": "src.models.baselines.nonaka.xresnet.xresnet1d50",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.models.sequence.rnns.sru.SRURNN",
    "lssl": "src.models.sequence.ss.lssl.LSSL",
    "s4": "src.models.sequence.ss.s4.S4",
    "standalone": "src.models.s4.s4.S4",
    "s4d": "src.models.s4.s4d.S4D",
    "ff": "src.models.sequence.ff.FF",
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "conv1d": "src.models.sequence.convs.conv1d.Conv1d",
    "conv2d": "src.models.sequence.convs.conv2d.Conv2d",
    "performer": "src.models.sequence.attention.linear.Performer",
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
}
