import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "som_quantize",
    version = "0.2.2",
    author = "Keir Havel",
    author_email = "keirhavel@live.com",
    description = "Pytorch implementation of vector quantization and self-organizing maps.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/LumenPallidium/quantization-maps",
    project_urls = {"Issue Tracker" : "https://github.com/LumenPallidium/quantization-maps/issues",
                    },
    license = "MIT",
    packages = ["som_quantize"],
    install_requires = ["torch>=2.0",
                        "einops"],
)