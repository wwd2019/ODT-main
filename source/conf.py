# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ODT-main'
copyright = '2026, Wendi Wang'
author = 'Wendi Wang'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx']

templates_path = ['_templates']
exclude_patterns = []
nbsphinx_allow_errors = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# 1. 设置 ReadTheDocs 官方主题
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# 2. 支持 Markdown 格式（通过 recommonmark）
from recommonmark.parser import CommonMarkParser
source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']  # 同时支持 .rst 和 .md 文件

# 3. （可选）如需启用更多 Markdown 高级功能，可添加 AutoStructify 配置
# 具体参考你笔记中提到的官方示例：https://github.com/rtfd/recommonmark/blob/master/docs/conf.py
