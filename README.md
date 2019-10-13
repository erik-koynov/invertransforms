[Documentation]: https:///gregunz.github.io/invertransforms/
[mail@gregunz.io]: mailto:mail@gregunz.io

[![](https://i.imgur.com/dFDH5Ro.jpg)](https://github.com/gregunz/invertransforms)

invertransforms
====

[![Build Status](https://img.shields.io/travis/com/gregunz/invertransforms.svg?style=for-the-badge)](https://travis-ci.com/gregunz/invertransforms)
[![Code Coverage](https://img.shields.io/codecov/c/gh/gregunz/invertransforms.svg?style=for-the-badge)](https://codecov.io/gh/gregunz/invertransforms)
[![pdoc3 on PyPI](https://img.shields.io/pypi/v/invertransforms.svg?color=blue&style=for-the-badge)](https://pypi.org/project/invertransforms)

A library which turns torchvision transformations __invertible__ and __replayable__.


Installation
------------
```bash
pip install invertransforms
```

Usage
-----
Simply replace previous torchvision import statements and enjoy the addons.

```python
# from torchvision import transforms as T
import invertransforms as T

transform = T.Compose([
  T.RandomCrop(size=256),
  T.ToTensor(),
])

img_tensor = transform(img)

# invert
img_again = transform.invert(img_tensor)

# replay
img_tensor2 = transform.replay(img2)
```
All transformations have an `inverse` transformation attached to it.


```python
inv_transform = transform.inverse()
img_inv = inv_transform(img)
```
__Notes:__

If a transformation is random, it is necessary to apply it once before calling `invert` or `inverse()`. Otherwise it will raise `InvertibleError`. 
On the otherhand, `replay` can be called before, it will simply set the randomness on its first call.


[Documentation]
-------------

The library's [documentation] contains the full list of [transformations](https://gregunz.github.io/invertransforms/#header-classes) which includes all the ones from torchvision and more.

Use Case
--------

This library can be particularly useful in following situations:
- Reverting your model outputs in order to stack predictions
- Applying the same transformations on two different images
- blah-blah


Features
--------
* Invert any transformations even random ones
* Replay any transformations even random ones
* All classes extends its torchvision transformation equivalent class
* Extensive unit testing
* blah-blah


Contribute
----------

You found a bug, think a feature is missing or just want to help ?

Please feel free to open an issue, pull request or contact me [mail@gregunz.io]


