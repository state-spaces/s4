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
}

model = {
    "model": "src.models.sequence.SequenceModel",
    "unet": "src.models.sequence.SequenceUNet",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "ckconv": "src.models.baselines.ckconv.ClassificationCKCNN",
    "unicornn": "src.models.baselines.unicornn.UnICORNN",
    "resnet": "src.models.baselines.resnet.ResnetSquare",
    "wrn": "src.models.baselines.resnet.WideResNet",
    "odelstm": "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.models.baselines.lipschitzrnn.RnnModels",
    "wavegan": "src.models.baselines.wavegan.WaveGANDiscriminator",
    "denseinception": "src.models.baselines.dense_inception.DenseInception",
    "sashimi": "src.models.sequence.sashimi.Sashimi",
    "wavenet": "src.models.baselines.wavenet.WaveNetModel",
    "samplernn": "src.models.baselines.samplernn.SampleRNN",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.models.sequence.rnns.sru.SRURNN",  # TODO not updated
    "lssl": "src.models.sequence.ss.lssl.LSSL",
    "s4": "src.models.sequence.ss.s4.S4",
    "standalone": "src.models.sequence.ss.standalone.s4.S4",
    "ff": "src.models.sequence.ff.FF",
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "mha": "src.models.sequence.mha.MultiheadAttention",
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
}
