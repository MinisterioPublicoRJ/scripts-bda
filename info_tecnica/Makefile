default:
	./save_file_pdf.sh

install:
	mkdir packages
	/opt/cloudera/parcels/Anaconda-5.0.1/bin/pip download -r requirements.txt -d packages/
	cd packages && ls *.tar.gz | xargs -L1 tar -xvzf
	cd packages && ls *.zip | xargs -L1 unzip -x
	cd packages && ls -d */ | xargs -n1 -I {} bash -c '../despython.sh {}'
	cd packages && unzip pure_sasl-0.6.2-py2-none-any.whl "*.py"
	cd packages && zip puresasl.zip puresasl/*

clean:
	rm -rf packages

