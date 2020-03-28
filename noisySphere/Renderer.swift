//
//  Renderer.swift
//  noisySphere
//
//  Created by Yupeng Gu on 3/20/19.
//  Copyright © 2019 Yupeng Gu. All rights reserved.
//

// Our platform independent renderer class

import Metal
import MetalKit
import MetalPerformanceShaders
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size & ~0xFF) + 0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    
    var screenRepTexture: MTLTexture?
    
    var waterDropTexture: MTLTexture
    var intermTexture: MTLTexture
    
    var convIntermTexture: MTLTexture
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotationMatrix: matrix_float4x4 = matrix_float4x4()
    
    let touchSemaphore = DispatchSemaphore(value: 1)
    
    var tOffset:Float = 0.0
    
    var threadGroupSize:MTLSize
    
    var threadGroupCount:MTLSize
    
    var threadGroupSizeThresh:MTLSize
    
    var threadGroupCountThresh:MTLSize
    
    var threadGroupSizeFire:MTLSize
    
    var threadGroupCountFire:MTLSize
    
    var computeNoisePipelineState:MTLComputePipelineState
    
    var computeNormalPipelineState:MTLComputePipelineState
    
    var computeThresholdPipelineState:MTLComputePipelineState
    
    var computeCombinePipelineState:MTLComputePipelineState
    
    var computeClearPipelineState:MTLComputePipelineState
    
    var computeDistortNoisePipelineState:MTLComputePipelineState
    
    var computeDistortPipelineState:MTLComputePipelineState
    
    var computeCombineFirePipelineState:MTLComputePipelineState
    
    var normalMap: MTLTexture
    
    var mesh: MTKMesh
    
    var metalLayer: CAMetalLayer
    
    var viewWidth:Int,viewHeight:Int
    
    init?(metalKitView: MTKView) {
        self.viewWidth = Int(metalKitView.drawableSize.width)
        self.viewHeight = Int(metalKitView.drawableSize.height)
        
        self.metalLayer = metalKitView.layer as! CAMetalLayer
        
        self.metalLayer.framebufferOnly = false
        
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        
        guard let buffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = buffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)
        
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm
        metalKitView.sampleCount = 1
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        do{
            computeNoisePipelineState = try Renderer.buildNoiseComputePipeline(device: device)
        }catch{
            print("Unable to compile compute pipeline state for noise.")
            return nil
        }
        
        do{
            computeNormalPipelineState = try Renderer.buildNormalComputePipeline(device: device)
        }catch{
            print("Unable to compile compute pipeline state for normal vectors.")
            return nil
        }
        
        do{
            computeThresholdPipelineState = try Renderer.buildThresholdComputePipeline(device: device)
        }catch{
            print("Unable to compile compute pipeline state for color threshold.")
            return nil
        }
        
        do{
            computeCombinePipelineState = try Renderer.buildCombineComputePipeline(device: device)
        }catch{
            print("Unable to compile compute pipeline state for color combine.")
            return nil
        }
        
        intermTexture = Renderer.buildIntermTexture(device: device)!
        
        do{
            computeClearPipelineState = try Renderer.buildClearComputePipeline(device: device)
        }catch {
            print("Unable to build clear compute pipeline state.")
            return nil
        }
        
        do {
            computeDistortNoisePipelineState = try Renderer.buildDistortNoiseComputePipeline(device: device)
        } catch {
            print("Unable to build distort noise compute pipeline state.")
            return nil
        }
        
        do{
            computeDistortPipelineState = try Renderer.buildDistortComputePipeline(device: device)
        }catch {
            print("Unable to build distort compute pipeline state.")
            return nil
        }
        
        do{
            computeCombineFirePipelineState = try Renderer.buildCombineFireComputePipeline(device: device)
        }catch {
            print("Unable to build combine fire compute pipeline state.")
            return nil
        }
        
        do {
            waterDropTexture = try Renderer.loadTexture(device: device, textureName: "Waterdrop")
        } catch {
            print("Unable to load texture waterdrop. Error info: \(error)")
            return nil
        }
        
        threadGroupSizeFire = MTLSize(width: 16, height: 16, depth: 1)
        threadGroupCountFire = MTLSize(width: 33, height: 33, depth: 1)
        
        threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        threadGroupCount = MTLSize(width: 16, height: 16, depth: 1)
        
        threadGroupSizeThresh = MTLSize(width: 16, height: 16, depth: 1)
        threadGroupCountThresh = MTLSize(width: viewWidth/threadGroupSizeThresh.width+1, height: viewHeight/threadGroupSizeThresh.height+1, depth: 1)
        
        
        let depthStateDesciptor = MTLDepthStencilDescriptor()
        depthStateDesciptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDesciptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDesciptor) else { return nil }
        depthState = state
        
        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        colorMap = Renderer.buildEmptyTexture(device: device)!
        
        normalMap = Renderer.buildEmptyTexture(device: device)!
        
        convIntermTexture = Renderer.buildConvIntermTexture(device: device, width: viewWidth, height: viewHeight)!
        
        super.init()
        
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Creete a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.meshNormals.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.tangent.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.tangent.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.tangent.rawValue].bufferIndex = BufferIndex.meshTagent.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.bitangent.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.bitangent.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.bitangent.rawValue].bufferIndex = BufferIndex.meshBitangent.rawValue
        
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshTagent.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshTagent.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshTagent.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshBitangent.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshBitangent.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshBitangent.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildClearComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "clearKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildDistortNoiseComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "distortNoiseKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildDistortComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "distortKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildNoiseComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "gaborNoiseKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildNormalComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "normalKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildThresholdComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "thresholdKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildCombineComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "combineKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildCombineFireComputePipeline(device:MTLDevice) throws -> MTLComputePipelineState{
        let library = device.makeDefaultLibrary()
        
        let kernelFunction = library?.makeFunction(name: "combineFireKernel")
        
        return try device.makeComputePipelineState(function: kernelFunction!)
    }
    
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.sampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        /*let mdlMesh = MDLMesh.newBox(withDimensions: float3(4, 4, 4),
                                     segments: uint3(2, 2, 2),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)*/
        let mdlMesh = MDLMesh.newEllipsoid(withRadii: float3(2,2,2), radialSegments: 32, verticalSegments: 32, geometryType: MDLGeometryType.triangles, inwardNormals: false, hemisphere: false, allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        attributes[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal
        attributes[VertexAttribute.tangent.rawValue].name = MDLVertexAttributeTangent
        attributes[VertexAttribute.bitangent.rawValue].name = MDLVertexAttributeBitangent
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    class func buildConvIntermTexture(device:MTLDevice,width:Int,height:Int) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = MTLTextureType.type3D
        textureDescriptor.pixelFormat = MTLPixelFormat.r8Unorm
        textureDescriptor.width = width
        textureDescriptor.height = height/2
        textureDescriptor.depth = 32
        textureDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue)
        
        return device.makeTexture(descriptor: textureDescriptor)
    }
    
    class func buildIntermTexture(device:MTLDevice) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = MTLTextureType.type2D
        textureDescriptor.pixelFormat = MTLPixelFormat.bgra8Unorm
        textureDescriptor.width = 513
        textureDescriptor.height = 513
        textureDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue)
        
        return device.makeTexture(descriptor: textureDescriptor)
    }
    
    func buildScreenRepTexture(device:MTLDevice) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = MTLTextureType.type2D
        textureDescriptor.pixelFormat = MTLPixelFormat.bgra8Unorm
        textureDescriptor.width = self.viewWidth
        textureDescriptor.height = self.viewHeight
        textureDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue)
        
        return device.makeTexture(descriptor: textureDescriptor)
    }
    
    class func buildEmptyTexture(device:MTLDevice) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = MTLTextureType.type2D
        textureDescriptor.pixelFormat = MTLPixelFormat.bgra8Unorm
        textureDescriptor.width = 256
        textureDescriptor.height = 256
        textureDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue)
        
        return device.makeTexture(descriptor: textureDescriptor)
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState() {
        /// Update any game state before rendering
        
        uniforms[0].projectionMatrix = projectionMatrix
        
        //let rotationAxis = float3(0, 1, 0)
        //let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -9.0)
        touchSemaphore.wait()
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, rotationMatrix)
        touchSemaphore.signal()
        uniforms[0].normalMatrix = simd_inverse(simd_transpose(uniforms[0].modelViewMatrix))
        uniforms[0].tOffset = tOffset
        tOffset += 0.04
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder(){
                computeEncoder.pushDebugGroup("Calculate Noise Map")
                
                computeEncoder.setComputePipelineState(computeNoisePipelineState)
                
                computeEncoder.setBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                computeEncoder.setTexture(colorMap, index: TextureIndex.output.rawValue)
                
                computeEncoder.dispatchThreadgroups(threadGroupSize, threadsPerThreadgroup: threadGroupCount)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.pushDebugGroup("Calculate Normal Map")
                
                computeEncoder.setComputePipelineState(computeNormalPipelineState)
                
                computeEncoder.setTexture(colorMap, index: TextureIndex.color.rawValue)
                
                computeEncoder.setTexture(normalMap, index: TextureIndex.output.rawValue)
                
                computeEncoder.dispatchThreadgroups(threadGroupSize, threadsPerThreadgroup: threadGroupCount)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.endEncoding()
            }
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            renderPassDescriptor?.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"
                
                renderEncoder.pushDebugGroup("Draw Box")
                
                renderEncoder.setCullMode(.back)
                
                renderEncoder.setFrontFacing(.counterClockwise)
                
                renderEncoder.setRenderPipelineState(pipelineState)
                
                renderEncoder.setDepthStencilState(depthState)
                
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                renderEncoder.setVertexTexture(colorMap, index: TextureIndex.color.rawValue)
                
                renderEncoder.setVertexTexture(normalMap, index: TextureIndex.output.rawValue)
                
                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        return
                    }
                    
                    if layout.stride != 0 {
                        let buffer = mesh.vertexBuffers[index]
                        renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                    }
                }
                
                renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
                
                for submesh in mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer,
                                                        indexBufferOffset: submesh.indexBuffer.offset)
                    
                }
                
                renderEncoder.popDebugGroup()
                
                renderEncoder.endEncoding()

            }
            
            let screenDrawable = view.currentDrawable!
            
            let screenTexture = screenDrawable.texture
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder(){
                
                computeEncoder.pushDebugGroup("Calculate Color Threshold")
                
                computeEncoder.setComputePipelineState(computeThresholdPipelineState)
                
                computeEncoder.setTexture(screenTexture, index: TextureIndex.color.rawValue)
                
                if screenRepTexture==nil{
                    screenRepTexture = buildScreenRepTexture(device: device)
                }
                
                computeEncoder.setTexture(screenRepTexture!, index: TextureIndex.output.rawValue)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountThresh, threadsPerThreadgroup: threadGroupSizeThresh)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.endEncoding()
                
                let blurKernel = MPSImageGaussianBlur(device: device, sigma: 50.0)
                
                blurKernel.encode(commandBuffer: commandBuffer, inPlaceTexture: &screenRepTexture!, fallbackCopyAllocator: nil)
            }
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder(){
                computeEncoder.pushDebugGroup("Clear kernel texture")
                
                computeEncoder.setComputePipelineState(computeClearPipelineState)
                
                computeEncoder.setTexture(intermTexture, index: 0)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountFire, threadsPerThreadgroup: threadGroupSizeFire)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.pushDebugGroup("Distort noise")
                
                computeEncoder.setComputePipelineState(computeDistortNoisePipelineState)
                
                computeEncoder.setTexture(waterDropTexture, index: 0)
                
                computeEncoder.setTexture(waterDropTexture, index: 1)
                
                computeEncoder.setBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountFire, threadsPerThreadgroup: threadGroupSizeFire)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.pushDebugGroup("Distort fire shape")
                
                computeEncoder.setComputePipelineState(computeDistortPipelineState)
                
                computeEncoder.setTexture(waterDropTexture, index: 0)
                
                computeEncoder.setTexture(intermTexture, index: 1)
                
                //computeEncoder.setBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountFire, threadsPerThreadgroup: threadGroupSizeFire)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.endEncoding()
            }
            
            let blurFireKernel = MPSImageGaussianBlur(device: device, sigma: 5.0)
            
            blurFireKernel.encode(commandBuffer: commandBuffer, inPlaceTexture: &intermTexture, fallbackCopyAllocator: nil)
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder(){
                
                computeEncoder.pushDebugGroup("Calculate Fire Color Combined")
                
                computeEncoder.setComputePipelineState(computeCombineFirePipelineState)
                
                computeEncoder.setTexture(screenTexture, index: 0)
                
                computeEncoder.setTexture(intermTexture, index: 1)
                
                computeEncoder.setTexture(screenTexture, index: 2)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountThresh, threadsPerThreadgroup: threadGroupSizeThresh)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.pushDebugGroup("Calculate Color Combined")
                
                computeEncoder.setComputePipelineState(computeCombinePipelineState)
                
                computeEncoder.setTexture(screenTexture, index: 0)
                
                if screenRepTexture==nil{
                    screenRepTexture = buildScreenRepTexture(device: device)
                }
                
                computeEncoder.setTexture(screenRepTexture!, index: 1)
                
                computeEncoder.setTexture(screenTexture, index: 2)
                
                computeEncoder.dispatchThreadgroups(threadGroupCountThresh, threadsPerThreadgroup: threadGroupSizeThresh)
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.endEncoding()
            }
            
            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
        rotationMatrix = matrix4x4_rotation(radians: 0.0, axis: float3(x: 0, y: 1, z: 0))
    }
    
    func rotate(radians: Float,axis: float3){
        DispatchQueue.global().async { [unowned self] in
            self.touchSemaphore.wait()
            let curRot = matrix4x4_rotation(radians: radians, axis: axis)
            self.rotationMatrix = simd_mul(self.rotationMatrix, curRot)
            self.touchSemaphore.signal()
        }
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: float3) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
