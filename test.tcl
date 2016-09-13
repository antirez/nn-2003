load tclgnegnu.so
set net [ann::create 42 2 42]
ann::configure net -algo rprop
set img1 {
    0 0 0 0 0 0 0
    0 0 1 1 1 0 0
    0 1 0 0 0 1 0
    0 1 0 0 0 1 0
    0 0 1 1 1 0 0
    0 0 0 0 0 0 0
}

set img2 {
    1 0 0 0 0 0 1
    1 1 0 1 1 1 0
    1 1 1 1 0 0 0
    1 1 1 1 0 0 0
    1 1 0 1 1 1 0
    1 0 0 0 0 0 1
}


set dataset [list $img1 $img1 $img2 $img2]

proc print img {
    set i 0
    for {set j 0} {$j < [llength $img]} {incr j} {
	if {[lindex $img $j] > 0.5} {
	    puts -nonewline "#"
	} else {
	    puts -nonewline "."
	}
	if {!(($j+1) % 7)} {
	    puts {}
	}
    }
}

print $img1
print $img2

puts [ann::train net $dataset 40]

set output [ann::simulate net $img1]
print $output
set output [ann::simulate net $img2]
print $output
