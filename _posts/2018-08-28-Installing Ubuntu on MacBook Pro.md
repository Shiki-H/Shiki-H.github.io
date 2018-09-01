---
layout:     post
title:      "Installing Ubuntu on MacBook Pro"
tags:
    - Linux
---

This posts records the steps I have taken to install Linux Mint 18.3 on MacBook Pro 11, 3 as well as some of the errors I have encoutnered with solutions I found online. The steps should be almost identical for installing Ubuntu.

## Install Linux Mint

Honestly this is the easiest step. In short, one just have to: 

- partition the disk in OS X using `Disk Utility`

- create a bootable USB stick  

- install Linux Mint 

More detailed steps can be found [here](http://www.daveoncode.com/2015/05/26/installing-and-configuring-linux-mint-in-dual-boot-on-an-apple-macbook-pro-with-retina-display/).

A few caveats regarding the tutorial given by the link above based on my experience:

- if you have windows machine, [Linux Live USB Creator](https://www.linuxliveusb.com/) can be a very handy tool to create bootable USB stick

- during installation, you may want to make sure you are connected to the internet. To do that, once you boot into the USB stick, open driver manager and install the wifi driver. This is similar to step 6.2 in the tutorial. 

## Install NVIDIA Drivers on MacBook Pro

Once you are done installing the system and set up internet connection, you can start working on this. The following steps should help you getting the latest official drivers from NVIDIA. First, open your terminal and enter

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

Then open drivers manager, you should be able to see the latest drivers available. Although it is possible to do all these in terminal, to save the trouble, I would recommend just using the drivers manager. 

Once you are done installing the drivers from drivers manager, the tough fight starts. 

### Adjust Brightness after Installing NVIDIA Driver

A common problem one encounters after installing NVIDIA drivers is not able to adjust brightness with function keys. To solve this problem, open your terminal, enter

```
sudo nano /etc/X11/xorg.conf
```

and copy the following

```bash
Section "Screen"
	Identifier	"Default Screen"
	DefaultDepth	24
EndSection
Section "Module"
	Load	"glx"
EndSection
Section "Device"
	Identifier	"Default Device"
	Driver	"nvidia"
	Option	"NoLogo"	"True"
	Option "RegistryDwords" "EnableBrightnessControl=1"
EndSection
```

Exit and save. Most tutorials would tell you a similar step when you are working with NVIDIA on Ubuntu. However, to make it work on MacBook, you need an extra step. Now, in the terminal, enter 

```bash
setpci -v -H1 -s 00:01.00 BRIDGE_CONTROL=0
```

Usually, it is a good idea to write this line to `/etc/rc.local` so it is executed when the system starts. 

### Using Intel Graphics Card in Ubuntu on MacBook Pro

Even if one has installed `nvidia-prime`, it is not possible to switch video card on a MacBook, because Intel's graphic card is not yet acessible to Ubuntu. To overcome this problem, we need a hack called `apple_set_os.efi` which is available [here](https://github.com/0xbb/apple_set_os.efi). 

Here are the steps to make it work on your MacBook adapted from the write-up [suggested by Andrey](https://github.com/Dunedan/mbp-2016-linux/issues/6).

#### 1. Build `apple_set_os.efi`

The author listed a few methods to build it. I would recommend build it from source and NOT use the relseased ones since they can be quirky. 

```bash
apt-get install gnu-efi
git clone https://github.com/0xbb/apple_set_os.efi
cd apple_set_os.efi
make
```

#### 2. Copy to EFI partition

```bash
sudo cp apple_set_os.efi /boot/efi/EFI/
```

#### 3. Add a new menu entry to Grub boot loader

Open terminal and enter

```bash
sudo nano /etc/grub.d/40_custom
```

and copy the following

```bash
menuentry "Unlock Intel IGD" {
  search --no-floppy --set=root --file /EFI/apple_set_os.efi
  chainloader /EFI/apple_set_os.efi
  boot
}
```

Then open `/etc/default/grub` file and make sure they look like 

```bash
GRUB_DEFAULT=0
# GRUB_HIDDEN_TIMEOUT=0
# GRUB_HIDDEN_TIMEOUT_QUIET=true
GRUB_TIMEOUT=10
GRUB_CMDLINE_LINUX="acpi_backlight=intel_backlight"
```

Now, run `sudo update-grub`

#### 4. Switch to Intel

```bash
git clone https://github.com/0xbb/gpu-switch
cd gpu-switch/
sudo ./gpu-switch -i
```

#### 5. Reboot

Once your MacBook reboot, before selecting the actual OS such Ubuntu or Mint, remember to select `Unlock Intel IGD` first. If not, you may see a flickering screen and have to for power down. 

Once you are in the system, you can verify if your Intel graphic is activated by

```bash
$ lspci | grep "VGA"
00:02.0 VGA compatible controller: Intel Corporation Crystal Well Integrated Graphics Controller (rev 08)
01:00.0 VGA compatible controller: NVIDIA Corporation GK107M [GeForce GT 750M Mac Edition] (rev a1)
$ glxinfo | grep "OpenGL renderer"
OpenGL renderer string: Mesa DRI Intel(R) Haswell Mobile
```

Once everythin works out correctly, you can skip the step of selecting `Unlock Intel IGD` by writing the following to `/etc/grub.d/40_custom`

```bash 
search --no-floppy --set=root --file /EFI/apple_set_os.efi
chainloader /EFI/apple_set_os.efi
boot
```

### Extra Steps for Overheating

The main motiavation behind using Intel graphics instead of NVIDIA is to reduce heat and enhance battery life. During my search online, I found a few other methods that could potentially help to reduce overheating. 

#### 1. `tlp`

TLP is a power management tool on Linux. To install, 

```bash 
sudo add-apt-repository ppa:linrunner/tlp
sudo apt-get update
sudo apt-get install tlp tlp-rdw
```

#### 2. `cpufreq`

This tool allows you to run your laptop in two modes, which are `performance` and `powersave`. I did not test how much energy it could save, just including it here for reference. 

```bash
sudo apt-get install indicator-cpufreq
```

Note that with newer Core processors, you will not be able to choose specific frequency with `cpufreq` as some screen shots you can find online. For more details, please refer to this [SE thread](https://askubuntu.com/questions/544266/why-are-missing-the-frequency-options-on-cpufreq-utils-indicator).

#### 3. `mbpfan`

This package uses the temperature from coretemp module as input and adjusts the fan speed accordingly. The installation is not very straightforward, but the author has provided a easy-to-follow step-by-step instruction [here](https://ineed.coffee/3838/a-beginners-tutorial-for-mbpfan-under-ubuntu/).

## References

http://www.daveoncode.com/2015/05/26/installing-and-configuring-linux-mint-in-dual-boot-on-an-apple-macbook-pro-with-retina-display/

https://iocrunch.com/2014/08/nvidia-backlight-in-linux-on-macbookpro-113/

https://github.com/Dunedan/mbp-2016-linux/issues/6

https://itsfoss.com/reduce-overheating-laptops-linux/

https://blog.csdn.net/redstone0001/article/details/17042011

https://askubuntu.com/questions/544266/why-are-missing-the-frequency-options-on-cpufreq-utils-indicator

https://ineed.coffee/3838/a-beginners-tutorial-for-mbpfan-under-ubuntu/
