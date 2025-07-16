# Things that you need to set up before installing BerryPicker

## TODO
* [ ] The data directories seem to be creating a directory called ~/... 
while this should be referring to the home directory. Why is this?
    * [ ] Maybe it is a bad character in the config file
    * [ ] Maybe the relative addressing is not working here?

## Python and venv
sudo apt install python3.10-venv
sudo apt install python-is-python3

## Sudoers
This is needed because of you need to set up the directory

sudo usermod -aG sudo al5d