
(define (domain None)
  (:requirements :typing)
  (:types )
  
  (:predicates (north_of ?v0 - r ?v1 - r)
	(edible ?v0 - f)
	(closed ?v0 - d)
	(locked ?v0 - c)
	(link ?v0 - r ?v1 - d ?v2 - r)
	(match ?v0 - k ?v1 - d)
	(on ?v0 - o ?v1 - s)
	(open ?v0 - d)
	(free ?v0 - r ?v1 - r)
	(in ?v0 - o ?v1 - I)
	(west_of ?v0 - r ?v1 - r)
	(at ?v0 - P ?v1 - r)
	(north_of/d ?v0 - r ?v1 - d ?v2 - r)
	(west_of/d ?v0 - r ?v1 - d ?v2 - r)
	(eaten ?v0 - f)
  )
  ; (:actions )

  

	(:action examine/t
		:parameters (P - P r - r t - t)
		:precondition (and (at P r)
			(at t r))
		:effect (and
			(at t r)
			(at P r))
	)
	

	(:action go/north
		:parameters (P - P r - r r' - r)
		:precondition (and (at P r)
			(north_of r' r)
			(free r r')
			(free r' r))
		:effect (and
			(north_of r' r)
			(free r r')
			(free r' r)
			(at P r'))
	)
	

	(:action go/south
		:parameters (P - P r - r r' - r)
		:precondition (and (at P r)
			(north_of r r')
			(free r r')
			(free r' r))
		:effect (and
			(north_of r r')
			(free r r')
			(free r' r)
			(at P r'))
	)
	

	(:action go/east
		:parameters (P - P r - r r' - r)
		:precondition (and (at P r)
			(west_of r r')
			(free r r')
			(free r' r))
		:effect (and
			(west_of r r')
			(free r r')
			(free r' r)
			(at P r'))
	)
	

	(:action go/west
		:parameters (P - P r - r r' - r)
		:precondition (and (at P r)
			(west_of r' r)
			(free r r')
			(free r' r))
		:effect (and
			(west_of r' r)
			(free r r')
			(free r' r)
			(at P r'))
	)
	

	(:action lock/c
		:parameters (P - P r - r c - c k - k I - I)
		:precondition (and (at P r)
			(at c r)
			(in k I)
			(match k c)
			(closed c))
		:effect (and
			(at P r)
			(at c r)
			(in k I)
			(match k c)
			(locked c))
	)
	

	(:action unlock/c
		:parameters (P - P r - r c - c k - k I - I)
		:precondition (and (at P r)
			(at c r)
			(in k I)
			(match k c)
			(locked c))
		:effect (and
			(at P r)
			(at c r)
			(in k I)
			(match k c)
			(closed c))
	)
	

	(:action open/c
		:parameters (P - P r - r c - c)
		:precondition (and (at P r)
			(at c r)
			(closed c))
		:effect (and
			(at P r)
			(at c r)
			(open c))
	)
	

	(:action close/c
		:parameters (P - P r - r c - c)
		:precondition (and (at P r)
			(at c r)
			(open c))
		:effect (and
			(at P r)
			(at c r)
			(closed c))
	)
	

	(:action inventory
		:parameters (P - P r - r)
		:precondition (and (at P r))
		:effect (and
			(at P r))
	)
	

	(:action take
		:parameters (P - P r - r o - o)
		:precondition (and (at P r)
			(at o r))
		:effect (and
			(at P r)
			(in o I))
	)
	

	(:action drop
		:parameters (P - P r - r o - o I - I)
		:precondition (and (at P r)
			(in o I))
		:effect (and
			(at P r)
			(at o r))
	)
	

	(:action take/c
		:parameters (P - P r - r c - c o - o)
		:precondition (and (at P r)
			(at c r)
			(open c)
			(in o c))
		:effect (and
			(at P r)
			(at c r)
			(open c)
			(in o I))
	)
	

	(:action insert
		:parameters (P - P r - r c - c o - o I - I)
		:precondition (and (at P r)
			(at c r)
			(open c)
			(in o I))
		:effect (and
			(at P r)
			(at c r)
			(open c)
			(in o c))
	)
	

	(:action take/s
		:parameters (P - P r - r s - s o - o)
		:precondition (and (at P r)
			(at s r)
			(on o s))
		:effect (and
			(at P r)
			(at s r)
			(in o I))
	)
	

	(:action put
		:parameters (P - P r - r s - s o - o I - I)
		:precondition (and (at P r)
			(at s r)
			(in o I))
		:effect (and
			(at P r)
			(at s r)
			(on o s))
	)
	

	(:action examine/I
		:parameters (o - o I - I)
		:precondition (and (in o I))
		:effect (and
			(in o I))
	)
	

	(:action examine/s
		:parameters (P - P r - r s - s o - o)
		:precondition (and (at P r)
			(at s r)
			(on o s))
		:effect (and
			(at s r)
			(on o s)
			(at P r))
	)
	

	(:action examine/c
		:parameters (P - P r - r c - c o - o)
		:precondition (and (at P r)
			(at c r)
			(open c)
			(in o c))
		:effect (and
			(at c r)
			(open c)
			(in o c)
			(at P r))
	)
	

	(:action look
		:parameters (P - P r - r)
		:precondition (and (at P r))
		:effect (and
			(at P r))
	)
	

	(:action eat
		:parameters (f - f I - I)
		:precondition (and (in f I))
		:effect (and
			(eaten f))
	)
	

	(:action lock/d
		:parameters (P - P r - r d - d r' - r k - k I - I)
		:precondition (and (at P r)
			(link r d r')
			(link r' d r)
			(in k I)
			(match k d)
			(closed d))
		:effect (and
			(at P r)
			(link r d r')
			(link r' d r)
			(in k I)
			(match k d)
			(locked d))
	)
	

	(:action unlock/d
		:parameters (P - P r - r d - d r' - r k - k I - I)
		:precondition (and (at P r)
			(link r d r')
			(link r' d r)
			(in k I)
			(match k d)
			(locked d))
		:effect (and
			(at P r)
			(link r d r')
			(link r' d r)
			(in k I)
			(match k d)
			(closed d))
	)
	

	(:action open/d
		:parameters (P - P r - r d - d r' - r)
		:precondition (and (at P r)
			(link r d r')
			(link r' d r)
			(closed d))
		:effect (and
			(at P r)
			(link r d r')
			(link r' d r)
			(open d)
			(free r r')
			(free r' r))
	)
	

	(:action close/d
		:parameters (P - P r - r d - d r' - r)
		:precondition (and (at P r)
			(link r d r')
			(link r' d r)
			(open d)
			(free r r')
			(free r' r))
		:effect (and
			(at P r)
			(link r d r')
			(link r' d r)
			(closed d))
	)
	

	(:action examine/d
		:parameters (P - P r - r d - d r' - r)
		:precondition (and (at P r)
			(link r d r'))
		:effect (and
			(link r d r')
			(at P r))
	)

  

)
        