import argparse
from hdfs import InsecureClient
import csv


def execute_process(args):

    directory = args.directory
    linux_path = args.linuxPath
    file_name = args.fileName
    end_file_name = file_name.replace(".csv", "_done.csv")

    full_path = '{}{}/'.format(linux_path, directory)

    full_file_name_path = '{}{}'.format(full_path, end_file_name)
    with open('{}{}'.format(full_path, file_name), 'rb') as read:
        with open(full_file_name_path, 'wb') as writer:
            reader = csv.reader(read, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                data = ';'.join(row)
                data = data.replace("\n", " ").replace("\r", " ") + "\n"
                writer.write(data)

    client = InsecureClient(args.webHdfs, args.userWebHdfs)
    client.upload(args.hdfsPath + directory, full_file_name_path, n_threads=5, overwrite=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Put file linux")
    parser.add_argument('-wh', '--webHdfs',
                        metavar='webHdfs', type=str, help='')
    parser.add_argument('-u', '--userWebHdfs',
                        metavar='userWebHdfs', type=str, help='')
    parser.add_argument('-d', '--directory',
                        metavar='directory', type=str, help='')
    parser.add_argument('-lp', '--linuxPath',
                        metavar='linuxPath', type=str, help='')
    parser.add_argument('-fn', '--fileName',
                        metavar='fileName', type=str, help='')
    parser.add_argument('-hp', '--hdfsPath',
                        metavar='hdfsPath', type=str, help='')

    args = parser.parse_args()
    execute_process(args)