default:
	spark-submit tabela_acervo.py

install:
	mkdir packages
	/opt/cloudera/parcels/Anaconda-5.0.1/bin/pip download -r requirements.txt -d packages/
	cd packages && ls *.tar.gz | xargs -L1 tar -xvzf
	cd packages && ls *.zip | xargs -L1 unzip -x
	cd packages && ls -d */ | xargs -n1 -I {} bash -c '../despython.sh {}'

clean:
	rm -rf packages

