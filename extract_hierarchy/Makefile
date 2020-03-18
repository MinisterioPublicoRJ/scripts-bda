default:
	./execute_extract_hierarchy.sh

install:
	mkdir packages
	/opt/cloudera/parcels/Anaconda-5.0.1/bin/pip download -r requirements.txt -d packages/
	cd packages && ls *.tar.gz | xargs -L1 tar -xvzf 
	cd packages && ls -d */ | xargs -n1 -I {} bash -c '../despython.sh {}'

clean:
	rm -rf packages

