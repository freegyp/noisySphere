//
//  ShaderTypes.h
//  noisySphere
//
//  Created by Yupeng Gu on 3/20/19.
//  Copyright © 2019 Yupeng Gu. All rights reserved.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

typedef NS_ENUM(NSInteger, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexMeshNormals   = 2,
    BufferIndexMeshTagent    = 3,
    BufferIndexMeshBitangent = 4,
    BufferIndexUniforms      = 5
};

typedef NS_ENUM(NSInteger, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
    VertexAttributeNormal = 2,
    VertexAttributeTangent = 3,
    VertexAttributeBitangent = 4,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexColor    = 0,
    TextureIndexOutput = 1
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 modelViewMatrix;
    matrix_float4x4 normalMatrix;
    float tOffset;
} Uniforms;

#endif /* ShaderTypes_h */

