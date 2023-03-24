# How/Where to Download all the data!

aria2c is the best linux utility I've found so far that can reliably download
a lot of data.

Here are the follow datasets that are supported currently and where to get them:

# R2N2 ShapeNet Dataset:
Size:

Contents:

Description:

# ShapeNetCore.v2:
Size:

Contents:

Description:

# Pix3D Dataset: 
Size:

Contents:

Description:

# ABO (Amazon Berkley Dataset):
Size:

Contents:

Description:



# Downloading Procedure:
This is owned and operated by ONR and MIT. You have to request an account (preferably using an `.edu` email.)

Once this is done though the link can be obtained link so:

and this should be added to the `aria_download_file.txt`


Once all the data links (https) is in the download text file, you can run it like so:
`aria2c -c -s 16 -x 16 -k 1M -j 1 -i aria_download_file.txt`

NOTE: This can download what some people to consider to be a very large amount of data, ***all unzipped it can be up 800+ GB***, so if you're only interested in using the dataloaders and modela for a specific file, you should comment out or remove the line of the datasets you aren't interested in from the `aria_download_file.txt`

