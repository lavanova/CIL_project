
if [ -d ./log ]; then
	echo "log directory already exist"
else
	mkdir log
fi


if [ -d ./log/model ]; then
	echo "log/model directory already exist"
else
	mkdir log/model
fi