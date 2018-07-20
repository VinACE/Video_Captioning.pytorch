ls -R amw > audio_list.txt
while read line 
do 
	echo $line
	python vggish_inference_demo.py --wav_file=./amw/$line --tfrecord_file=./h5s/$line.h5 --pca_params=vggish_pca_params.npz --checkpoint=vggish_model.ckpt
done <audio_list.txt
