default:
	./example_create_process_spark.sh

install:
	mkdir packages
	/opt/cloudera/parcels/Anaconda-5.0.1/bin/pip download -r requirements.txt -d packages/
	cd packages && ls *.tar.gz | xargs -L1 tar -xvzf 
	cd packages && ls -d */ | xargs -n1 -I {} bash -c '../despython.sh {}'
	cd packages && ls *.whl | xargs -n 1 unzip -x
	cd packages && ls *.egg | xargs -n 1 unzip -x


clean:
	rm -rf packages
