"""
Author: Jose L Balcazar, ORCID 0000-0003-4248-4528 (feb 2017)
Copyleft: MIT License (https://en.wikipedia.org/wiki/MIT_License)

Ignore this file if you want to contribute to Suverat.

One correct usage of Popeel.

Spoiler. 

Don't read this code.

Hidden in a different project (Suverat) and in some Dropbox so as to
avoid its visibility from the standard Popeel users.
"""

from popeel import Popeel

p = Popeel()
p.set_task(12)
while not p.enough_potatoes():
	if p.basket_is_empty():
		p.refill_basket()
	p.peel_1_potato()
p.discard_basket()
p.go_sleep()
