mkdir -p dataset/test/
wget https://raw.github.com/circulosmeos/gdown.pl/master/gdown.pl
chmod u+x gdown.pl
./gdown.pl 'https://drive.google.com/file/d/1Ee08vMIC_Rtm1AGukFV7xbcFEwqgwRF0/view?usp=sharing' dataset/test/AgeDB-aligned.tar.gz
./gdown.pl 'https://drive.google.com/file/d/1r8ND8smGFeqsrmeO7d6fZcdYU62-kLzf/view?usp=sharing' dataset/test/FGNET-aligned.tar.gz
./gdown.pl 'https://drive.google.com/file/d/1yY9npRtK-U8DeM-p-FfiIWMYVXGzbJyy/view?usp=sharing' dataset/test/LAG-aligned.tar.gz
tar -zxvf dataset/test/AgeDB-aligned.tar.gz --directory dataset/test/
tar -zxvf dataset/test/FGNET-aligned.tar.gz --directory dataset/test/
tar -zxvf dataset/test/LAG-aligned.tar.gz --directory dataset/test/

rm -rf dataset/test/AgeDB-aligned.tar.gz
rm -rf dataset/test/FGNET-aligned.tar.gz
rm -rf dataset/test/LAG-aligned.tar.gz
rm -rf gdown.pl

mv dataset/test/AgeDB_new_align dataset/test/AgeDB-aligned
mv dataset/test/FGNET_new_align dataset/test/FGNET-aligned
mv dataset/test/LAG_new_align dataset/test/LAG-aligned
