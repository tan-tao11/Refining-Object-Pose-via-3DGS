set -e

DATASET_FOLDER=$1

cd $DATASET_FOLDER

for dir in *; do
    #delete if file
    if [ -f "$dir" ]; then
        rm -rf dir
    fi

    mv $dir model_ship-${dir##*-}
done

for dir in *; do
    mv $dir/box3d_corners .
    break
done

mkdir model-ship
mv * model-ship