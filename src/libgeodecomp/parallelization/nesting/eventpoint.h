#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_EVENTPOINT_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_EVENTPOINT_H

enum EventPoint {LOAD_BALANCING, END};
typedef std::set<EventPoint> EventSet;
typedef std::map<long, EventSet> EventMap;

#endif
