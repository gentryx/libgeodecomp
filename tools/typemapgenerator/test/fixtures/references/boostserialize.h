    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, Car& object, const unsigned /*version*/)
    {
        archive & object.engine;
        archive & object.wheels;
    }

