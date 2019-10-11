using CSV, DataFrames
using VideoIO, Images #, Makie
using Plots
using Flux
using Flux: @epochs

FRAME_LEN = 512
TRAIN_FN = "./data/train.mp4"
TEST_FN = "./data/test.mp4"
LABELS_FN = "./data/train.txt"


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

io, vid = main()
speeds = get_speeds()

m = Chain(Dense(640*480, 1000, tanh), Dense(1000, 1))

function loss(x, y)
	img = view(Gray.(x), :)
	l = Flux.mse(m(img), y)
	return l
end

function make_tups(vid, speeds)
	i = 1
	tups = []
	while !eof(vid)
		img = read(vid)
		speed = speeds[i]
		push!(tups, (img, speed))
		i += 1
	end
	return tups
end
tups = make_tups(vid, speeds)

ps = Flux.params(m)
loss(tups[1]...)
plot(tups[1][1])

opt = ADAM()
num_tups = length(tups)

function evalcb()
	test_x, test_y = tups[rand(1:num_tups)]
	img = view(Gray.(test_x), :)
	plot!(test_x)
	@show(loss(test_x, test_y))
	@show(m(img), test_y)
end
Flux.train!(loss, ps, tups, opt, cb = Flux.throttle(evalcb, 5))
# run()
