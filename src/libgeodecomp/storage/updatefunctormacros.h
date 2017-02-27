#ifndef LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H
#define LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H

#include <libgeodecomp/storage/updatefunctormacrosmsvc.h>

#ifndef _MSC_BUILD

#ifdef LIBGEODECOMP_WITH_THREADS
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1                         \
    if (concurrencySpec.enableOpenMP() &&                               \
        !modelThreadingSpec.hasOpenMP()) {                              \
        if (concurrencySpec.preferStaticScheduling()) {                 \
            _Pragma("omp parallel for schedule(static)")                \
            for (std::size_t c = 0; c < region.numPlanes(); ++c) {      \
                typename Region<DIM>::StreakIterator e =                \
                    region.planeStreakIterator(c + 1);                  \
                typedef typename Region<DIM>::StreakIterator Iter;      \
                for (Iter i = region.planeStreakIterator(c + 0);        \
                     i != e;                                            \
                     ++i) {                                             \
                    LGD_UPDATE_FUNCTOR_BODY;                            \
                }                                                       \
            }                                                           \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2                         \
        } else {                                                        \
            if (!concurrencySpec.preferFineGrainedParallelism()) {      \
                _Pragma("omp parallel for schedule(dynamic)")           \
                for (std::size_t c = 0; c < region.numPlanes(); ++c) {  \
                    typename Region<DIM>::StreakIterator e =            \
                        region.planeStreakIterator(c + 1);              \
                    typedef typename Region<DIM>::StreakIterator Iter;  \
                    for (Iter i = region.planeStreakIterator(c + 0);    \
                         i != e;                                        \
                         ++i) {                                         \
                        LGD_UPDATE_FUNCTOR_BODY;                        \
                    }                                                   \
                }                                                       \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3                         \
            } else {                                                    \
                typedef typename Region<DIM>::StreakIterator Iter;      \
                std::vector<Streak<DIM> > streaks;                      \
                streaks.reserve(region.numStreaks());                   \
                for (Iter i = region.beginStreak();                     \
                     i != region.endStreak();                           \
                     ++i) {                                             \
                    Streak<DIM> s = *i;                                 \
                    auto granularity = modelThreadingSpec.granularity(); \
                    while (s.length() > granularity) {                  \
                        Streak<DIM> tranche = s;                        \
                        tranche.endX = s.origin.x() + granularity -     \
                            (s.origin.x() % granularity);               \
                        streaks.push_back(tranche);                     \
                        s.origin.x() = tranche.endX;                    \
                    }                                                   \
                    streaks.push_back(s);                               \
                }                                                       \
                _Pragma("omp parallel for schedule(dynamic)")           \
                    for (std::size_t j = 0; j < streaks.size(); ++j) {  \
                        Streak<DIM> *i = &streaks[j];                   \
                        LGD_UPDATE_FUNCTOR_BODY;                        \
                    }                                                   \
            }                                                           \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4                         \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5                         \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6                         \
        }                                                               \
        return;                                                         \
    }
#else
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
#endif

#endif

#ifdef LIBGEODECOMP_WITH_HPX
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7                         \
    if (concurrencySpec.enableHPX() && !modelThreadingSpec.hasHPX()) {  \
        if (!concurrencySpec.preferFineGrainedParallelism()) {          \
            std::vector<hpx::future<void> > updateFutures;              \
            updateFutures.reserve(region.numPlanes());                  \
            typedef typename Region<DIM>::StreakIterator Iter;          \
            Iter last = region.beginStreak();                           \
                                                                        \
            for (std::size_t i = 0; i < region.numPlanes(); ++i) {      \
                Iter next = region.planeStreakIterator(i + 1);          \
                updateFutures << hpx::async(                            \
                            [&](Iter i, Iter end) {                     \
                                for(; i != end; ++i) {                  \
                                    LGD_UPDATE_FUNCTOR_BODY;            \
                                }                                       \
                            }, last, next);                             \
                last = next;                                            \
            }                                                           \
                                                                        \
            hpx::wait_all(updateFutures);                               \
            return;                                                     \
        } else {                                                        \
            std::vector<hpx::future<void> > updateFutures;              \
            typedef typename Region<DIM>::StreakIterator Iter;          \
                                                                        \
            std::vector<Streak<DIM> > streaks;                          \
            streaks.reserve(region.numStreaks());                       \
                                                                        \
            for (Iter i = region.beginStreak();                         \
                 i != region.endStreak();                               \
                 ++i) {                                                 \
                Streak<DIM> s = *i;                                     \
                auto granularity = modelThreadingSpec.granularity();    \
                while (s.length() > granularity) {                      \
                    Streak<DIM> tranche = s;                            \
                    tranche.endX = s.origin.x() + granularity -         \
                        (s.origin.x() % granularity);                   \
                    streaks.push_back(tranche);                         \
                    s.origin.x() = tranche.endX;                        \
                }                                                       \
                streaks.push_back(s);                                   \
            }                                                           \
                                                                        \
            updateFutures.reserve(streaks.size());                      \
                                                                        \
            for (auto& streak: streaks) {                               \
                updateFutures << hpx::async(                            \
                    [&](Streak<DIM> *i) {                               \
                        LGD_UPDATE_FUNCTOR_BODY;                        \
                    }, &streak);                                        \
            }                                                           \
                                                                        \
            hpx::wait_all(updateFutures);                               \
            return;                                                     \
        }                                                               \
    }                                                                   \
    /**/
#else
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
    /**/
#endif

#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8                         \
    for (typename Region<DIM>::StreakIterator i = region.beginStreak(); \
         i != region.endStreak();                                       \
         ++i) {                                                         \
        LGD_UPDATE_FUNCTOR_BODY;                                        \
    }                                                                   \
    /**/


#endif
