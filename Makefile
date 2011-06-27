first:	compile

compile:
	rake

test: compile
	rake test

install:
	rake install

clean:
	rake clean

distclean:
	rake distclean

.PHONY : compile test install clean distclean


