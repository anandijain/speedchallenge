using CSV, DataFrames

using VideoIO, Makie


FRAME_LEN = 512 
TRAIN_FN = "train.mp4"
TEST_FN = "test.mp4"
LABELS_FN = "train.txt"


function get_speeds()
	df = CSV.read(LABELS_FN, header=["mph"])
	speeds = convert(Vector, df.mph)
	return speeds
end

function main()
	io = VideoIO.open(TRAIN_FN)
	vid = VideoIO.openvideo(io)
	return io, vid
end

function run()
	io, vid = main()
	speeds = get_speeds()
	img = read(vid)
	scene = Makie.Scene(resolution = size(img))
	makieimg = Makie.image!(scene, img)[end]
	Makie.rotate!(scene, -0.5pi)
	display(scene)
	i = 1  
	while !eof(vid)
		println(speeds[i])
    		read!(vid, img)
    		makieimg[1] = img
    		sleep(1/vid.framerate)
		i += 1

	end
end	


run()
