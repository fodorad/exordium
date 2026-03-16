from exordium.video.face.blink.densenet import BlinkDenseNet121Wrapper as BlinkDenseNet121

# For backwards compatibility, alias BlinkDenseNet121 as BlinkWrapper
BlinkWrapper = BlinkDenseNet121

__all__ = ["BlinkDenseNet121", "BlinkWrapper"]
