from torchvision import transforms

from invertransforms.util import Invertible, InvertibleError


class Compose(transforms.Compose, Invertible):
    def inverse(self):
        transforms_inv = []
        for t in self.transforms[::-1]:
            if not isinstance(t, Invertible):
                raise InvertibleError(f'{t} ({t.__class__.__name__}) is not an invertible object')
            transforms_inv.append(t.inverse())
        return Compose(transforms=transforms_inv)
