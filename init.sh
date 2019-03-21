#! /bin/bash

if [ -d ./data ]; then
	echo "Data directory already exist"
else
	mkdir data
fi

if [ -d ./cache ]; then
	echo "Cache directory already exist"
else
	mkdir cache
fi
