# makefile for easy manage package
.PHONY: clean
clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "MANIFEST" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	rm -rf limetr MRTool

install_env:
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh && \
		conda create -n $(env_name) python=3.7 -y && conda activate $(env_name) && \
		pip install numpy scipy pandas && \
		conda install -c conda-forge cyipopt gmp h5py -y && \
		pip install --global-option=build_ext --global-option "-I$(CONDA_PREFIX)/envs/$(env_name)/include/" pycddlib && \
		git clone https://github.com/zhengp0/limetr.git && \
		git clone https://github.com/ihmeuw-msca/MRTool.git && \
		cd limetr && git checkout master && make install && cd .. && \
		cd MRTool && python setup.py install && cd .. && \
		pip install -e .[internal]

.PHONY: test
test:
	pytest -vv tests
