# Placeholder de TFT. Implementación completa requiere dataloaders, escaladores y configuración avanzada.
# Se deja como stub para mantener la interfaz.
class TFTModel:
    def __init__(self, **kwargs):
        self.cfg = kwargs
    def fit(self, *args, **kwargs):
        return self
    def predict(self, *args, **kwargs):
        raise NotImplementedError("Implementa TFT con pytorch-forecasting o usa ARIMAX/XGBoost por ahora.")
