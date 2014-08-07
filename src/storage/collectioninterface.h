#ifndef LIBGEODECOMP_STORAGE_COLLECTIONINTERFACE_H
#define LIBGEODECOMP_STORAGE_COLLECTIONINTERFACE_H

namespace LibGeoDecomp {

/**
 * This class allows library code to interface with various collection
 * classes (e.g. arrays, lists, user models). Its purpose is to bridge
 * the different interfaces to one, minimal interface.
 */
class CollectionInterface
{
public:
    /**
     * This is basically a non-operate: begin() and end() are passed
     * on to the consumer of this class.
     */
    template<
        typename CELL,
        typename CARGO = typename CELL::value_type,
        typename ITERATOR = typename CELL::iterator,
        typename CONST_ITERATOR = typename CELL::const_iterator>
    class PassThrough
    {
    public:
        // CARGO corresponds to value_type in std::vector and friends.
        typedef CARGO Cargo;
        typedef ITERATOR Iterator;
        typedef CONST_ITERATOR ConstIterator;
        typedef CELL Container;

        CELL& operator()(CELL& cell) const
        {
            return cell;
        }

        const CELL& operator()(const CELL& cell) const
        {
            return cell;
        }

        ITERATOR begin(CELL& cell) const
        {
            return cell.begin();
        }

        CONST_ITERATOR begin(const CELL& cell) const
        {
            return cell.begin();
        }

        ITERATOR end(CELL& cell) const
        {
            return cell.end();
        }

        CONST_ITERATOR end(const CELL& cell) const
        {
            return cell.end();
        }

        std::size_t size(const CELL& cell) const
        {
            return cell.size();
        }
    };

    /**
     * This can be used to delegate calls to a member of the given
     * CELL -- useful if your CELL contains multiple collections, e.g.
     * particles of different type, or nodes and elements.
     *
     * This is not unlike the Selector class template, which we use to
     * select single member variables.
     */
    template<
        typename CELL,
        typename CONTAINER,
        typename ITERATOR = typename CONTAINER::iterator,
        typename CONST_ITERATOR = typename CONTAINER::const_iterator>
    class Delegate
    {
    public:
        // CARGO corresponds to value_type in std::vector and friends.
        typedef typename CONTAINER::value_type Cargo;
        typedef ITERATOR Iterator;
        typedef CONST_ITERATOR ConstIterator;

        Delegate(CONTAINER CELL:: *memberPointer) :
            memberPointer(memberPointer)
        {}

        ITERATOR begin(CELL& cell) const
        {
            return (cell.*memberPointer).begin();
        }

        CONST_ITERATOR begin(const CELL& cell) const
        {
            return (cell.*memberPointer).begin();
        }

        ITERATOR end(CELL& cell) const
        {
            return (cell.*memberPointer).end();
        }

        CONST_ITERATOR end(const CELL& cell) const
        {
            return (cell.*memberPointer).end();
        }

        std::size_t size(const CELL& cell) const
        {
            return (cell.*memberPointer).size();
        }

    private:
        CONTAINER CELL:: *memberPointer;
    };

};


}

#endif
