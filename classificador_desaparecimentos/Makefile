default:
	./save_file_pdf.sh

install:
	mkdir packages
	/opt/cloudera/parcels/Anaconda-5.0.1/bin/pip download -r requirements.txt -d packages/
	cd packages && ls *.tar.gz | xargs -L1 tar -xvzf 
	cd packages && ls -d */ | xargs -n1 -I {} bash -c 'sh ../despython.sh {}'
	cd packages && ls -d */ | xargs rm -rf

clean:
	rm -rf packages

