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
    "model": "src.models.sequence.SequenceModel",
    "unet": "src.models.sequence.SequenceUNet",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "convnet": "src.models.sequence.convnet",
    "ckconv": "src.models.baselines.ckconv.ClassificationCKCNN",
    "unicornn": "src.models.baselines.unicornn.UnICORNN",
    "resnet": "src.models.baselines.resnet.ResnetSquare",
    "wrn": "src.models.baselines.resnet.WideResNet",
    "odelstm": "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.models.baselines.lipschitzrnn.RnnModels",
    "wavegan": "src.models.baselines.wavegan.WaveGANDiscriminator",
    "denseinception": "src.models.baselines.dense_inception.DenseInception",
    "stackedrnn": "src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline": "src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn": "src.models.baselines.samplernn.SampleRNN",
    "dcgru": "src.models.baselines.dcgru.DCRNNModel_classification",
    "dcgru_ss": "src.models.baselines.dcgru.DCRNNModel_nextTimePred",
    "vit": "models.baselines.vit.ViT",
    "snet": "src.models.sequence.snet.SequenceSNet",
    "sashimi": "src.models.sequence.sashimi.Sashimi",
    "wavenet": "src.models.baselines.wavenet.WaveNetModel",
    # ViT Variants (note: small variant is taken from Tri, differs from original)
    "vit_s_16": "src.models.baselines.vit_all.vit_small_patch16_224",
    "vit_b_16": "src.models.baselines.vit_all.vit_base_patch16_224",
    "convnext": "src.models.baselines.convnext.convnext",
    "convnext_timm_base": "src.models.baselines.convnext_timm.convnext_base",
    "convnext_timm_small": "src.models.baselines.convnext_timm.convnext_small",
    "convnext_timm_tiny": "src.models.baselines.convnext_timm.convnext_tiny",
    "convnext_timm_micro": "src.models.baselines.convnext_timm.convnext_micro",
    "convnext_timm_orig": "src.models.baselines.convnext_timm_orig.convnext_base",
    "convnext_timm_orig_v0": "src.models.baselines.convnext_timm_orig_v0.convnext_base",
    "resnet50_timm": "src.models.baselines.resnet_timm.resnet50",
    "s4nd_unet": "src.models.sequence.unet_nd.S4NDUNet",
    "convnext_timm_tiny_3d": "src.models.baselines.convnext_timm.convnext3d_tiny",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.models.sequence.rnns.sru.SRURNN",  # TODO not updated
    "lssl": "src.models.sequence.ss.lssl.LSSL",
    "s4": "src.models.sequence.ss.s4.S4",
    "standalone": "src.models.sequence.ss.standalone.s4.S4",
    "s4d": "src.models.sequence.ss.standalone.s4d.S4D",
    "s4_2d": "src.models.sequence.ss.s4_2d.StateSpace2D",
    "s4nd": "src.models.sequence.ss.s4_nd.S4ND",
    "ff": "src.models.sequence.ff.FF",
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "conv1d": "src.models.sequence.conv1d.Conv1d",
    "attsimp": "src.models.sequence.mha.AttentionSimple",
    "performer": "src.models.sequence.attention.linear.Performer",
    "s4_2dconv": "src.models.sequence.ss.s4_2dconv.S42DConv",
    "s4_simple": "src.models.sequence.ss.s4_simple.s4_wrapper.SimpleS4Wrapper",
    # 'packedrnn': 'models.sequence.rnns.packedrnn.PackedRNN',
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
    "progressive_learning": "src.callbacks.progressive_learning.ProgressiveLearning",
}

layer_decay = {
    'convnext_timm_tiny': 'src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny',
}

model_state_hook = {
    'convnext_timm_tiny_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d',
    'convnext_timm_tiny_s4nd_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d',
}
