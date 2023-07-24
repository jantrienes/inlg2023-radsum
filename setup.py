import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="guidedsum",
    version="0.0.1",
    author="Jan Trienes",
    author_email="jan.trienes@uni-due.de",
    description="Guidance in Radiology Report Summarization: An Empirical Evaluation and Error Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jantrienes/inlg2023-radsum",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.6",
)
