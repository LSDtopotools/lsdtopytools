// $Id$
//
//  sha1.h
//
//  Copyright (C) 1998
//  Paul E. Jones <paulej@arid.us>
//  All Rights Reserved.
//
//  This software is licensed as "freeware."  Permission to distribute
//  this software in source and binary forms is hereby granted without
//  a fee.  THIS SOFTWARE IS PROVIDED 'AS IS' AND WITHOUT ANY EXPRESSED
//  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//  THE AUTHOR SHALL NOT BE HELD LIABLE FOR ANY DAMAGES RESULTING
//  FROM THE USE OF THIS SOFTWARE, EITHER DIRECTLY OR INDIRECTLY, INCLUDING,
//  BUT NOT LIMITED TO, LOSS OF DATA OR DATA BEING RENDERED INACCURATE.
//
//////////////////////////////////////////////////////////////////////////////
//  ###Id: sha1.h,v 1.6 2004/03/27 18:02:26 paulej Exp ###
//  Modified March 7, 2007 - Andy Tompkins - change to a header only library
//////////////////////////////////////////////////////////////////////////////
//
//  Description:
//      This class implements the Secure Hashing Standard as defined
//      in FIPS PUB 180-1 published April 17, 1995.
//
//      Many of the variable names in this class, especially the single
//      character names, were used because those were the names used
//      in the publication.
//
//      The Secure Hashing Standard, which uses the Secure Hashing
//      Algorithm (SHA), produces a 160-bit message digest for a
//      given data stream.  In theory, it is highly improbable that
//      two messages will produce the same message digest.  Therefore,
//      this algorithm can serve as a means of providing a "fingerprint"
//      for a message.
//
//  Portability Issues:
//      SHA-1 is defined in terms of 32-bit "words".  This code was
//      written with the expectation that the processor has at least
//      a 32-bit machine word size.  If the machine word size is larger,
//      the code should still function properly.  One caveat to that
//      is that the input functions taking characters and character arrays
//      assume that only 8 bits of information are stored in each character.
//
//  Caveats:
//      SHA-1 is designed to work with messages less than 2^64 bits long.
//      Although SHA-1 allows a message digest to be generated for
//      messages of any number of bits less than 2^64, this implementation
//      only works with messages with a length that is a multiple of 8
//      bits.
//
#ifndef LIBLAS_SHA1_HPP_INCLUDED
#define LIBLAS_SHA1_HPP_INCLUDED

#include <cassert>
#include <boost/array.hpp>

namespace liblas { namespace detail {

class SHA1
{
public:
    /*  
     *  SHA1
     *
     *  Description:
     *      This is the constructor for the sha1 class.
     *
     *  Parameters:
     *      None.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    SHA1()
    {
        Reset();
    }

    /*  
     *  ~SHA1
     *
     *  Description:
     *      This is the destructor for the sha1 class
     *
     *  Parameters:
     *      None.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    ~SHA1()
    {
        // The destructor does nothing
    }
    
    /*  
     *  Reset
     *
     *  Description:
     *      This function will initialize the sha1 class member variables
     *      in preparation for computing a new message digest.
     *
     *  Parameters:
     *      None.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void Reset()
    {
        Length_Low          = 0;
        Length_High         = 0;
        Message_Block_Index = 0;

        H[0]        = 0x67452301;
        H[1]        = 0xEFCDAB89;
        H[2]        = 0x98BADCFE;
        H[3]        = 0x10325476;
        H[4]        = 0xC3D2E1F0;

        Computed    = false;
        Corrupted   = false;
    }
    
    /*  
     *  Result
     *
     *  Description:
     *      This function will return the 160-bit message digest into the
     *      array provided.
     *
     *  Parameters:
     *      message_digest_array: [out]
     *          This is an array of five unsigned integers which will be filled
     *          with the message digest that has been computed.
     *
     *  Returns:
     *      True if successful, false if it failed.
     *
     *  Comments:
     *
     */
    bool Result(::boost::array<unsigned int, 5>& message_digest_array)
    {
        int i = 0; // Counter
    
        if (Corrupted)
        {
            return false;
        }
    
        if (!Computed)
        {
            PadMessage();
            Computed = true;
        }
    
        for(i = 0; i < 5; i++)
        {
            message_digest_array[i] = H[i];
        }
    
        return true;
    }
    
    void Input(::boost::array<unsigned char, 16> message_array)
    {
        Input(message_array.c_array(), message_array.size());
    }

    /*  
     *  Input
     *
     *  Description:
     *      This function accepts an array of octets as the next portion of
     *      the message.
     *
     *  Parameters:
     *      message_array: [in]
     *          An array of characters representing the next portion of the
     *          message.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void Input(unsigned char const* message_array, std::size_t length)
    {
        if (0 == message_array || 0 == length)
        {
            return;
        }
    
        if (Computed || Corrupted)
        {
            Corrupted = true;
            return;
        }
    
        while(length-- && !Corrupted)
        {
            Message_Block[Message_Block_Index++] = (*message_array & 0xFF);
    
            Length_Low += 8;
            Length_Low &= 0xFFFFFFFF;               // Force it to 32 bits
            if (Length_Low == 0)
            {
                Length_High++;
                Length_High &= 0xFFFFFFFF;          // Force it to 32 bits
                if (Length_High == 0)
                {
                    Corrupted = true;               // Message is too long
                }
            }
    
            if (Message_Block_Index == 64)
            {
                ProcessMessageBlock();
            }
    
            message_array++;
        }
    }
    
    /*  
     *  Input
     *
     *  Description:
     *      This function accepts an array of octets as the next portion of
     *      the message.
     *
     *  Parameters:
     *      message_array: [in]
     *          An array of characters representing the next portion of the
     *          message.
     *      length: [in]
     *          The length of the message_array
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void Input(char const* message_array, unsigned length)
    {
        assert(0 != message_array);
        assert(0 != length);

        typedef unsigned char const* target_type;
        typedef void const* proxy_type;

        target_type p = static_cast<target_type>(static_cast<proxy_type>(message_array));
        Input(p, length);
    }
    
    /*  
     *  Input
     *
     *  Description:
     *      This function accepts a single octets as the next message element.
     *
     *  Parameters:
     *      message_element: [in]
     *          The next octet in the message.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void Input(unsigned char message_element)
    {
        Input(&message_element, 1);
    }
    
    /*  
     *  Input
     *
     *  Description:
     *      This function accepts a single octet as the next message element.
     *
     *  Parameters:
     *      message_element: [in]
     *          The next octet in the message.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void Input(char message_element)
    {
        unsigned char ch = static_cast<unsigned char>(message_element);
        Input(&ch, 1);
    }
    
    /*  
     *  operator<<
     *
     *  Description:
     *      This operator makes it convenient to provide character strings to
     *      the SHA1 object for processing.
     *
     *  Parameters:
     *      message_array: [in]
     *          The character array to take as input.
     *
     *  Returns:
     *      A reference to the SHA1 object.
     *
     *  Comments:
     *      Each character is assumed to hold 8 bits of information.
     *
     */
    SHA1& operator<<(const char *message_array)
    {
        assert(0 != message_array);

        const char* p = message_array;

        while(0 != p && *p)
        {
            Input(*p);
            p++;
        }

        return *this;
    }
    
    /*  
     *  operator<<
     *
     *  Description:
     *      This operator makes it convenient to provide character strings to
     *      the SHA1 object for processing.
     *
     *  Parameters:
     *      message_array: [in]
     *          The character array to take as input.
     *
     *  Returns:
     *      A reference to the SHA1 object.
     *
     *  Comments:
     *      Each character is assumed to hold 8 bits of information.
     *
     */
    SHA1& operator<<(const unsigned char *message_array)
    {
        assert(0 != message_array);

        const unsigned char *p = message_array;
    
        while(0 != p && *p)
        {
            Input(*p);
            p++;
        }
    
        return *this;
    }
    
    /*  
     *  operator<<
     *
     *  Description:
     *      This function provides the next octet in the message.
     *
     *  Parameters:
     *      message_element: [in]
     *          The next octet in the message
     *
     *  Returns:
     *      A reference to the SHA1 object.
     *
     *  Comments:
     *      The character is assumed to hold 8 bits of information.
     *
     */
    SHA1& operator<<(const char message_element)
    {
        Input((unsigned char *) &message_element, 1);
    
        return *this;
    }
    
    /*  
     *  operator<<
     *
     *  Description:
     *      This function provides the next octet in the message.
     *
     *  Parameters:
     *      message_element: [in]
     *          The next octet in the message
     *
     *  Returns:
     *      A reference to the SHA1 object.
     *
     *  Comments:
     *      The character is assumed to hold 8 bits of information.
     *
     */
    SHA1& operator<<(const unsigned char message_element)
    {
        Input(&message_element, 1);
    
        return *this;
    }
    
private:
    /*  
     *  ProcessMessageBlock
     *
     *  Description:
     *      This function will process the next 512 bits of the message
     *      stored in the Message_Block array.
     *
     *  Parameters:
     *      None.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *      Many of the variable names in this function, especially the single
     *      character names, were used because those were the names used
     *      in the publication.
     *
     */
    void ProcessMessageBlock()
    {
        const unsigned K[] =
        {               // Constants defined for SHA-1
            0x5A827999,
            0x6ED9EBA1,
            0x8F1BBCDC,
            0xCA62C1D6
        };

        int t = 0; // Loop counter
        unsigned int temp = 0; // Temporary word value
        unsigned int W[80]; // Word sequence
        unsigned int A = 0; // Word buffers
        unsigned int B = 0;
        unsigned int C = 0;
        unsigned int D = 0;
        unsigned int E = 0;

        /*
        *  Initialize the first 16 words in the array W
        */
        for(t = 0; t < 16; t++)
        {
            W[t] = ((unsigned) Message_Block[t * 4]) << 24;
            W[t] |= ((unsigned) Message_Block[t * 4 + 1]) << 16;
            W[t] |= ((unsigned) Message_Block[t * 4 + 2]) << 8;
            W[t] |= ((unsigned) Message_Block[t * 4 + 3]);
        }

        for(t = 16; t < 80; t++)
        {
            W[t] = CircularShift(1,W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
        }

        A = H[0];
        B = H[1];
        C = H[2];
        D = H[3];
        E = H[4];

        for(t = 0; t < 20; t++)
        {
            temp = CircularShift(5,A) + ((B & C) | ((~B) & D)) + E + W[t] + K[0];
            temp &= 0xFFFFFFFF;
            E = D;
            D = C;
            C = CircularShift(30,B);
            B = A;
            A = temp;
        }

        for(t = 20; t < 40; t++)
        {
            temp = CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[1];
            temp &= 0xFFFFFFFF;
            E = D;
            D = C;
            C = CircularShift(30,B);
            B = A;
            A = temp;
        }

        for(t = 40; t < 60; t++)
        {
            temp = CircularShift(5,A) +
                ((B & C) | (B & D) | (C & D)) + E + W[t] + K[2];
            temp &= 0xFFFFFFFF;
            E = D;
            D = C;
            C = CircularShift(30,B);
            B = A;
            A = temp;
        }

        for(t = 60; t < 80; t++)
        {
            temp = CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[3];
            temp &= 0xFFFFFFFF;
            E = D;
            D = C;
            C = CircularShift(30,B);
            B = A;
            A = temp;
        }

        H[0] = (H[0] + A) & 0xFFFFFFFF;
        H[1] = (H[1] + B) & 0xFFFFFFFF;
        H[2] = (H[2] + C) & 0xFFFFFFFF;
        H[3] = (H[3] + D) & 0xFFFFFFFF;
        H[4] = (H[4] + E) & 0xFFFFFFFF;

        Message_Block_Index = 0;
    }
    
    /*  
     *  PadMessage
     *
     *  Description:
     *      According to the standard, the message must be padded to an even
     *      512 bits.  The first padding bit must be a '1'.  The last 64 bits
     *      represent the length of the original message.  All bits in between
     *      should be 0.  This function will pad the message according to those
     *      rules by filling the message_block array accordingly.  It will also
     *      call ProcessMessageBlock() appropriately.  When it returns, it
     *      can be assumed that the message digest has been computed.
     *
     *  Parameters:
     *      None.
     *
     *  Returns:
     *      Nothing.
     *
     *  Comments:
     *
     */
    void PadMessage()
    {
        /*
        *  Check to see if the current message block is too small to hold
        *  the initial padding bits and length.  If so, we will pad the
        *  block, process it, and then continue padding into a second block.
        */
        if (Message_Block_Index > 55)
        {
            Message_Block[Message_Block_Index++] = 0x80;
            while(Message_Block_Index < 64)
            {
                Message_Block[Message_Block_Index++] = 0;
            }

            ProcessMessageBlock();

            while(Message_Block_Index < 56)
            {
                Message_Block[Message_Block_Index++] = 0;
            }
        }
        else
        {
            Message_Block[Message_Block_Index++] = 0x80;
            while(Message_Block_Index < 56)
            {
                Message_Block[Message_Block_Index++] = 0;
            }
        }

        /*
        *  Store the message length as the last 8 octets
        */
        Message_Block[56] = static_cast<unsigned char>((Length_High >> 24) & 0xFF);
        Message_Block[57] = static_cast<unsigned char>((Length_High >> 16) & 0xFF);
        Message_Block[58] = static_cast<unsigned char>((Length_High >> 8) & 0xFF);
        Message_Block[59] = static_cast<unsigned char>((Length_High) & 0xFF);
        Message_Block[60] = static_cast<unsigned char>((Length_Low >> 24) & 0xFF);
        Message_Block[61] = static_cast<unsigned char>((Length_Low >> 16) & 0xFF);
        Message_Block[62] = static_cast<unsigned char>((Length_Low >> 8) & 0xFF);
        Message_Block[63] = static_cast<unsigned char>((Length_Low) & 0xFF);

        ProcessMessageBlock();
    }

    /*  
     *  CircularShift
     *
     *  Description:
     *      This member function will perform a circular shifting operation.
     *
     *  Parameters:
     *      bits: [in]
     *          The number of bits to shift (1-31)
     *      word: [in]
     *          The value to shift (assumes a 32-bit integer)
     *
     *  Returns:
     *      The shifted value.
     *
     *  Comments:
     *
     */
    unsigned CircularShift(int bits, unsigned word)
    {
        return ((word << bits) & 0xFFFFFFFF) | ((word & 0xFFFFFFFF) >> (32-bits));
    }
    
private:

    boost::array<unsigned int, 5> H;    // Message digest buffers

    unsigned Length_Low;                // Message length in bits
    unsigned Length_High;               // Message length in bits

    boost::array<unsigned char, 64> Message_Block; // 512-bit message blocks
    int Message_Block_Index;            // Index into message block array

    bool Computed;                      // Is the digest computed?
    bool Corrupted;                     // Is the message digest corruped?
};

}} //namespace liblas::detail
    
#endif // LIBLAS_SHA1_HPP_INCLUDED
