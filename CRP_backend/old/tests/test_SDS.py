
###Test TODO
data, target = SDS.get_data_sample(10)
data1, target1 = SDS.DMI.get_data_sample(10)

print((data == data1).all())

data, target = SDS.get_data_sample(3233)
data2 = SDS.extra_DIs[0].get_data_sample_no_target(200)

print((data == data2).all())

datas, targets = SDS.get_data_concurrently([10, 3233])

print((datas[0] == data1).all())
print((datas[1] == data2).all())

SDS.set_analysis_mode("lol")

data, target = SDS.get_data_sample(10*200)
data1, target1 = SDS.DMI.get_data_sample(10)

print((data == data1).all())

data, target = SDS.get_data_sample(3233*200)
data2 = SDS.extra_DIs[0].get_data_sample_no_target(200)

print((data == data2).all())

datas, targets = SDS.get_data_concurrently([10*200, 3233*200])

print((datas[0] == data1).all())
print((datas[1] == data2).all())

file_name1, datasetname = SDS.data_index_to_filename(3233*200)
file_name_ori = SDS.extra_DIs[0].data_index_to_filename(200)
print(file_name_ori == file_name1)

SDS.set_analysis_mode("efficient")
index = SDS.data_filename_to_index(file_name1, datasetname)

print(index == 3233)

file_name2, datasetname = SDS.data_index_to_filename(3233)
index = SDS.data_filename_to_index(file_name2, datasetname)

print(index == 3233)
print(file_name1 == file_name2)

SDS.set_analysis_mode("effiasdsdcient")
file_name1, datasetname = SDS.data_index_to_filename(10*200)
file_name_ori = SDS.DMI.data_index_to_filename(10)
print(file_name_ori == file_name1)

SDS.set_analysis_mode("efficient")
index = SDS.data_filename_to_index(file_name1, datasetname)

print(index == 10)

file_name2, datasetname = SDS.data_index_to_filename(10)
index = SDS.data_filename_to_index(file_name2, datasetname)

print(index == 10)
print(file_name1 == file_name2)
#####