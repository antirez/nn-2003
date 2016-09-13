package require Tk
load tclgnegnu.so

proc reduce l {
    set ylen [llength $l]
    for {set y 0} {$y < $ylen} {incr y 2} {
	set scanline {}
	foreach {p _} [lindex $l $y] {
	    lappend scanline $p
	}
	lappend result $scanline
    }
    return $result
}

proc reduced2net l {
    scan2net [reduce $l]
}

proc scan2net l {
    set l [eval concat $l]
    foreach p $l {
	lappend nl [expr {double([scan $p #%02x])/255}]
    }
    return $nl
}

proc net2scan {l bxlen bylen} {
    set c 0
    for {set i 0} {$i < $bylen} {incr i} {
	set scanline {}
	for {set j 0} {$j < $bxlen} {incr j} {
	    set p [lindex $l $c]
	    incr c
	    set p [expr {int($p*255)}]
	    lappend scanline [format "#%02x%02x%02x" $p $p $p]
	}
	lappend result $scanline
    }
    return $result
}

proc blocks {ximglen yimglen bxlen bylen} {
    for {set y 0} {$y < $yimglen} {incr y $bylen} {
	for {set x 0} {$x < $ximglen} {incr x $bxlen} {
	    lappend result [list $x $y [expr {$x+$bxlen}] [expr {$y+$bylen}]]
	}
    }
    return $result
}

set blockxlen 8
set blockylen 8
set blockpixels [expr {$blockylen*$blockxlen}]
pack [canvas .c -height 1024 -width 1024]

set i [image create photo -file lena.pgm]
set i2 [image create photo -file peppers.pgm]
set io [image create photo -file lena.pgm]
set i2o [image create photo -file peppers-scaled.pgm]
set imgid [.c create image 256 256 -image $i]
set imgid2 [.c create image 256 768 -image $i2]
set imgid3 [.c create image 768 256 -image $io]
set imgid4 [.c create image 768 768 -image $i2o]
update idletasks

set blocks [blocks 512 512 $blockxlen $blockylen]
set blocks2 [blocks 256 256 4 4]
set net [ann::create $blockpixels 16 16]

# Create the dataset
foreach b $blocks {
    foreach {x1 y1 x2 y2} $b break
    set bi [reduced2net [$i data -grayscale -from $x1 $y1 $x2 $y2]]
    set bo [scan2net [$i data -grayscale -from $x1 $y1 $x2 $y2]]
    lappend dataset $bi $bo
}

# Create the second dataset
foreach b $blocks2 {
    foreach {x1 y1 x2 y2} $b break
    set bi [scan2net [$i2o data -grayscale -from $x1 $y1 $x2 $y2]]
    lappend dataset2 $bi $bo
}

# Train the network
while {1} {
    puts .
    ann::train net $dataset 25
    set c 0
    foreach b $blocks {
	foreach {x1 y1 x2 y2} $b break
	set netinput [lindex $dataset $c]
	incr c 2
	set netoutput [ann::simulate net $netinput]
	set scanlines [net2scan $netoutput $blockxlen $blockylen]
	$i put $scanlines -to $x1 $y1 $x2 $y2
    }
    set c 0
    foreach b $blocks2 bo $blocks {
	foreach {x1 y1 x2 y2} $b break
	foreach {ox1 oy1 ox2 oy2} $bo break
	set netinput [lindex $dataset2 $c]
	incr c 2
	set netoutput [ann::simulate net $netinput]
	set scanlines [net2scan $netoutput $blockxlen $blockylen]
	$i2 put $scanlines -to $ox1 $oy1 $ox2 $oy2
    }
    update idletasks
}

#set l [$i data -grayscale -from 0 0 8 8]
#set ni [scan2net $l]
#puts $ni
#puts [net2scan $ni]
exit
