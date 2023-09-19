import argparse
import struct

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.i8bin)")
    parser.add_argument("--label", help="The input label file (.i8bin)")
    parser.add_argument("--dst", help="The output file (.i8bin)")
    parser.add_argument("--num", type=int, help="file number")
    parser.add_argument("--cluster", type=int, help="cluster number")
    return parser.parse_args()

if __name__ == "__main__":
    args = process_args()

    assigned_cluster = []

    for i in range(0, args.cluster):
        empty_list = []
        assigned_cluster.append(empty_list)

    for i in range(0, args.num):
        #process file by file

        label_name = args.label + '.' + str(i)
        with open(label_name, "rb") as f_label:
            print(label_name)
            row_bin = ""
            dim_bin = ""
            row_bin = f_label.read(4)
            assert row_bin != b''
            row, = struct.unpack('i', row_bin)

            dim_bin = f_label.read(4)
            assert dim_bin != b''
            dim, = struct.unpack('i', dim_bin)

            for j in range(0, row):
                for k in range (0, dim):
                    cluster, = struct.unpack('h', f_label.read(2))
                    if cluster < args.cluster:
                        assigned_cluster[cluster].append(j + row * i)

    # print(assigned_cluster)

    for i in range(0, args.cluster):
        cluster_mapping_name = args.dst + '.label.' + str(i)
        length = len(assigned_cluster[i])
        print(length)
        with open(cluster_mapping_name, "wb") as f_mapping:
            f_mapping.write(struct.pack('i', length))
            f_mapping.write(struct.pack('i', 1))
            for j in range(0, length):
                f_mapping.write(struct.pack('i', assigned_cluster[i][j]))

    # for i in range(0, args.cluster):
    #     cluster_mapping_name = args.dst + '.label.' + str(i)
    #     with open(cluster_mapping_name, "rb") as f_mapping:
    #         row_bin = f_mapping.read(4)
    #         assert row_bin != b''
    #         row, = struct.unpack('i', row_bin)

    #         dim_bin = f_mapping.read(4)
    #         for j in range(0, row):
    #             assigned_cluster[i].append(struct.unpack('i', f_mapping.read(4)))
            

    for i in range(0, args.cluster):
        cluster_name = args.dst + '.' + str(i)
        with open(cluster_name, "wb") as f_cluster:
            with open(args.src, "rb") as f_origin:
                row_bin = ""
                dim_bin = ""
                row_bin = f_origin.read(4)
                assert row_bin != b''
                row, = struct.unpack('i', row_bin)

                dim_bin = f_origin.read(4)
                assert dim_bin != b''
                dim, = struct.unpack('i', dim_bin)

                index = 0
                last = 0
                length = len(assigned_cluster[i])
                print(cluster_name)
                print(length)
                f_cluster.write(struct.pack('i', length))
                f_cluster.write(dim_bin)

                while index != length:
                    f_origin.read(dim * (assigned_cluster[i][index] - last))
                    last = assigned_cluster[i][index] + 1
                    index+=1
                    f_cluster.write(f_origin.read(dim))


            

            



