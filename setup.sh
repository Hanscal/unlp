# 打包上传到PyPI
# 运行 python setup.py bdist_wheel (生成wheel文件）
python setup.py bdist_wheel
# 运行命令上传到PyPI
twine upload dist/*