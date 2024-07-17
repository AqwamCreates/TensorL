local AqwamTensorLibrary = {}

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function getSubTensorLength(tensor, targetDimension)

	local numberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(tensor)

	if (numberOfDimensions == targetDimension) then return #tensor end

	return getSubTensorLength(tensor[1], targetDimension)

end

local function applyFunctionUsingOneTensor(operation, tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local tensor = {}

	for i = 1, #tensor do  

		if (#dimensionSizeArray > 1) then

			tensor[i] = applyFunctionUsingOneTensor(operation, tensor[i])

		else

			tensor[i] = operation(tensor[i])

		end

	end

	return tensor
	
end

local function applyFunctionUsingTwoTensors(operation, tensor1, tensor2)

	local dimensionSizeArray1 = AqwamTensorLibrary:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary:getSize(tensor2)

	for i, _ in ipairs(dimensionSizeArray1) do if (dimensionSizeArray1[i] ~= dimensionSizeArray2[i]) then error("Invalid dimensions.") end end

	local tensor = {}

	for i = 1, #tensor1 do  

		if (#dimensionSizeArray1 > 1) then

			tensor[i] = applyFunctionUsingTwoTensors(operation, tensor1[i], tensor2[i])

		else

			tensor[i] = operation(tensor1[i], tensor2[i])

		end

	end

	return tensor

end

local function generateTensorString(tensor, dimensionDepth)

	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local result = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		result = result .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do

			if (i > 1) then result = result .. spacing end

			result = result .. generateTensorString(tensor[i], dimensionDepth + 1)

			if (i == tensorLength) then continue end

			result = result .. "\n"

		end

		result = result .. " }"

	else

		result = result .. "{ "

		for i = 1, tensorLength do 

			result = result .. tensor[i]

			if (i == tensorLength) then continue end

			result = result .. " "

		end

		result = result .. " }"

	end

	return result

end

local function generateTensorStringWithComma(tensor, dimensionDepth)
	
	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local result = " "

	if (numberOfDimensions > 1) then
		
		local spacing = ""

		result = result .. "{"
		
		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do
			
			if (i > 1) then result = result .. spacing end

			result = result .. generateTensorStringWithComma(tensor[i], dimensionDepth + 1)

			if (i == tensorLength) then continue end

			result = result .. "\n"

		end

		result = result .. " }"

	else

		result = result .. "{ "

		for i = 1, tensorLength do 

			result = result .. tensor[i]
		
			if (i == tensorLength) then continue end

			result = result .. ", "

		end

		result = result .. " }"

	end

	return result

end

local function generatePortableTensorString(tensor, dimensionDepth)

	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local result = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		result = result .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do

			if (i > 1) then result = result .. spacing end

			result = result .. generatePortableTensorString(tensor[i], dimensionDepth + 1)

			if (i == tensorLength) then continue end

			result = result .. "\n"

		end

		result = result .. " }"
		
		if (dimensionDepth > 1) then result = result .. "," end

	else

		result = result .. "{ "

		for i = 1, tensorLength do 

			result = result .. tensor[i]

			if (i == tensorLength) then continue end

			result = result .. ", "

		end

		result = result .. " },"

	end

	return result

end

local function fullSum(tensor)

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfValues = dimensionArray[1]

	local result = 0

	for i = 1, numberOfValues, 1 do 

		if (#dimensionArray > 1) then

			result += fullSum(tensor[i]) 

		else

			result += tensor[i]

		end

	end

	return result

end

local function dimensionSumRecursive(result, tensor, dimension)

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionArray

	local numberOfValues = dimensionArray[1]

	for i = 1, numberOfValues, 1 do 	

		if (numberOfDimensions == dimension) then

			dimensionSumRecursive(result[i], tensor[i], dimension)

		else

			result[i] += tensor[i]

		end

	end

end

local function accumulateDimension(result, tensor, dimension, dimensionIndices, dimensionArray)
	if dimension == #dimensionIndices then
		for i = 1, dimensionArray[dimensionIndices[dimension]], 1 do
			local indices = {}
			for j = 1, #dimensionIndices do
				indices[j] = dimensionIndices[j]
			end
			indices[dimension] = i
			result[table.unpack(indices)] = (result[table.unpack(indices)] or 0) + tensor[table.unpack(indices)]
		end
	else
		for i = 1, dimensionArray[dimensionIndices[dimension]], 1 do
			dimensionIndices[dimension] = i
			accumulateDimension(result, tensor, dimension + 1, dimensionIndices, dimensionArray)
		end
	end
end

local function dimSumRecursive(result, tensor, targetDimension)

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local currentDimension = #dimensionArray

	local numberOfValues = dimensionArray[1]

	for i = 1, numberOfValues, 1 do

		if (currentDimension == targetDimension) then

			print(AqwamTensorLibrary:getSize(result))

			print(AqwamTensorLibrary:getSize(tensor))

			result[i] += tensor[i]

		else

			dimensionSumRecursive(result[i], tensor[i], targetDimension)

		end

	end

end

local function dimensionSum(tensor, targetDimension)

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local newDimensionArray = deepCopyTable(dimensionArray)

	dimensionArray[targetDimension] = 1

	local result = createTensor(dimensionArray, 0)

	dimSumRecursive(result, tensor, targetDimension)

	--[[

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do

				if (dimension == 1) then

					result[1][dimension2][dimension3] += tensor[dimension1][dimension2][dimension3]	

				elseif (dimension == 2) then

					result[dimension1][1][dimension3] += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 3) then

					result[dimension1][dimension2][1] += tensor[dimension1][dimension2][dimension3]

				else

					error("Invalid dimension.")

				end 

			end

		end	

	end
	
	--]]

	return result

end

function AqwamTensorLibrary:sum(tensor, dimension)

	if (not dimension) then return fullSum(tensor) end

	local numberOfDimension = AqwamTensorLibrary:getNumberOfDimensions(tensor)

	if (dimension > numberOfDimension) or (dimension < 1) then error("Invalid dimensions.") end

	local reversedSequence = {}

	for i = numberOfDimension, 1, -1 do table.insert(reversedSequence, i) end

	local selectedDimension = reversedSequence[dimension]

	return dimensionSum(tensor, selectedDimension)

end

local function tensorProduct(tensor1, tensor2)

	local dimensionArray1 = AqwamTensorLibrary:getSize(tensor1)

	local dimensionArray2 = AqwamTensorLibrary:getSize(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]

	local result = {}

	for i = 1, numberOfValues, 1 do

		if (#dimensionArray1 > 1) then

			local subproduct = tensorProduct(tensor1[i], tensor2[i])

			table.insert(result, subproduct)

		else

			table.insert(result, tensor1[i] * tensor2[i])

		end

	end

	return result
end

local function truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	while true do

		local size = dimensionSizeArray[1]

		if (size ~= 1) then break end

		table.remove(dimensionSizeArray, 1)

	end

	return dimensionSizeArray

end

local function truncateTensorIfRequired(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (dimensionSizeArray[1] == 1) then

		return truncateTensorIfRequired(tensor[1])

	else

		return tensor

	end 

end


local function containFalseBooleanInTensor(booleanTensor)

	local dimensionArray = AqwamTensorLibrary:getSize(booleanTensor)

	local numberOfValues = dimensionArray[1]

	local result = true

	if (#dimensionArray > 1) then

		for i = 1, numberOfValues do result = containFalseBooleanInTensor(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 

			result = (result == booleanTensor[i])

			if (not result) then return false end

		end

	end

	return result

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local firstTensor = tensorArray[1]

	if (numberOfTensors == 1) then 
		
		local tensor = applyFunctionUsingOneTensor(functionToApply, firstTensor)
		
		tensor = truncateTensorIfRequired(tensor)
		
		return tensor
		
	end

	local tensor = firstTensor

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		tensor, otherTensor = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor, otherTensor)

		tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor)

	end
	
	tensor = truncateTensorIfRequired(tensor)

	return tensor

end

local function onBroadcastError(dimensionSizeArray1, dimensionSizeArray2)
	
	local numberOfDimensions1 = #dimensionSizeArray1
	
	local numberOfDimensions2 = #dimensionSizeArray2
	
	local tensor1DimensionSizeArrayString = "("
	
	local tensor2DimensionSizeArrayString = "("
	
	for s, size in ipairs(dimensionSizeArray1) do
		
		tensor1DimensionSizeArrayString = tensor1DimensionSizeArrayString .. size
		
		if (s ~= numberOfDimensions1) then
			
			tensor1DimensionSizeArrayString = tensor1DimensionSizeArrayString .. ", "
			
		end
		
	end
	
	for s, size in ipairs(dimensionSizeArray2) do

		tensor2DimensionSizeArrayString = tensor2DimensionSizeArrayString .. size

		if (s ~= numberOfDimensions2) then

			tensor2DimensionSizeArrayString = tensor2DimensionSizeArrayString .. ", "

		end

	end
	
	tensor1DimensionSizeArrayString = tensor1DimensionSizeArrayString .. ")"
	
	tensor2DimensionSizeArrayString = tensor2DimensionSizeArrayString .. ")"

	local errorMessage = "Unable To Broadcast. \n" .. "Tensor 1 Size: " .. tensor1DimensionSizeArrayString .."\n" .. "Tensor 2 Size: " .. tensor2DimensionSizeArrayString .."\n"
	
	error(errorMessage)

end

function AqwamTensorLibrary:expand1(tensor, targetDimensionSizeArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local tensorNumberOfDimensions = #dimensionSizeArray
	
	local targetNumberOfDimensions = #targetDimensionSizeArray
	
	if (targetNumberOfDimensions <= tensorNumberOfDimensions) then error("Unable to expand. The target number of dimensions is less than or equal to the tensor's number of dimensions.") end
	
	for i = 1, tensorNumberOfDimensions, 1 do -- We need to remove the extra dimensions from target dimension size array. The values are removed starting from the first so that we can compare the endings.

		table.remove(targetDimensionSizeArray, 1)

		if (#targetDimensionSizeArray == tensorNumberOfDimensions) then break end

	end
	
	for i = 1, tensorNumberOfDimensions, 1 do -- Check if the endings are equal so that we can expand the tensor. If the endings are not equal, then we can't expand the tensor.

		if (targetDimensionSizeArray[i] ~= dimensionSizeArray[i]) then error("Unable to expand. Different size at index " .. i) end

	end
	
end

function AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)
	
	local expandedTensor = {}

	if (#dimensionSizeToAddArray > 1) then
		
		local remainingDimensionSizeArray = {}

		for i = 2, #dimensionSizeToAddArray, 1 do table.insert(remainingDimensionSizeArray, dimensionSizeToAddArray[i]) end
		
		for i = 1, #dimensionSizeToAddArray, 1 do expandedTensor[i] = AqwamTensorLibrary:expand(tensor, remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeToAddArray[1], 1 do

			expandedTensor[i] = deepCopyTable(tensor)

		end

	end
	
	return expandedTensor
	
end

function AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor1, tensor2)
	
	local dimensionSizeArray1 = AqwamTensorLibrary:getSize(tensor1)
	
	local dimensionSizeArray2 = AqwamTensorLibrary:getSize(tensor2)
	
	local numberOfDimensions1 = #dimensionSizeArray1 
	
	local numberOfDimensions2 = #dimensionSizeArray2
	
	local haveSameNumberOfDimensions = (numberOfDimensions1 == numberOfDimensions2)
	
	if (haveSameNumberOfDimensions) then -- If the number of dimensions are equal, we need to make sure that the sizes in each dimensions are equal, so that we can return the tensors in their own original sizes.
		
		for s, size in ipairs(dimensionSizeArray1) do if (size ~= dimensionSizeArray2[s]) then onBroadcastError(dimensionSizeArray1, dimensionSizeArray2) end end
		
		return tensor1, tensor2
		
	end
	
	local isTensor1HaveLessNumberOfDimensions = (numberOfDimensions1 < numberOfDimensions2)
	
	local tensorNumberWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and 1) or 2
	
	local tensorWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and tensor1) or tensor2
	
	local dimensionSizeArrayWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and dimensionSizeArray1) or dimensionSizeArray2
	
	local dimensionSizeArrayWithHighestNumberOfDimensions = ((not isTensor1HaveLessNumberOfDimensions) and dimensionSizeArray1) or dimensionSizeArray2
	
	local copyOfDimensionSizeArrayWithHighestNumberOfDimensions = table.clone(dimensionSizeArrayWithHighestNumberOfDimensions)
	
	local lowestNumberOfDimensions = #dimensionSizeArrayWithLowestNumberOfDimensions
	
	local highestNumberOfDimensions = #dimensionSizeArrayWithHighestNumberOfDimensions
	
	local numberOfDimensionDifferences = highestNumberOfDimensions - lowestNumberOfDimensions
	
	for i = 1, lowestNumberOfDimensions, 1 do -- We need to remove the extra dimensions from tensor with highest number of dimensions. The values are removed starting from the first so that we can compare the endings.
		
		table.remove(copyOfDimensionSizeArrayWithHighestNumberOfDimensions, 1)
		
		if (#copyOfDimensionSizeArrayWithHighestNumberOfDimensions == lowestNumberOfDimensions) then break end
		
	end
	
	for i = 1, lowestNumberOfDimensions, 1 do -- Check if the endings are equal so that we can broadcast one of the tensor. If the endings are not equal, then we can't broadcast the tensor with the lowest number of dimensions.
		
		if (copyOfDimensionSizeArrayWithHighestNumberOfDimensions[i] ~= dimensionSizeArrayWithLowestNumberOfDimensions[i]) then onBroadcastError(dimensionSizeArray1, dimensionSizeArray2) end
		
	end
	
	local dimensionSizeToAdd = {}
	
	for i = 1, numberOfDimensionDifferences, 1 do
		
		table.insert(dimensionSizeToAdd, dimensionSizeArrayWithHighestNumberOfDimensions[i])
		
	end
	
	local expandedTensor = AqwamTensorLibrary:expand(tensorWithLowestNumberOfDimensions, dimensionSizeToAdd)
	
	if (tensorNumberWithLowestNumberOfDimensions == 1) then
		
		return expandedTensor, tensor2
		
	else
		
		return tensor1, expandedTensor
		
	end
	
end

function AqwamTensorLibrary:createTensor(dimensionSizeArray, initialValue)
	
	initialValue = initialValue or 0
	
	dimensionSizeArray = truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	local tensor = {}

	if (#dimensionSizeArray > 2) then

		local remainingDimensionSizeArray = {}

		for i = 2, #dimensionSizeArray, 1 do table.insert(remainingDimensionSizeArray, dimensionSizeArray[i]) end

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = AqwamTensorLibrary:createTensor(remainingDimensionSizeArray, initialValue) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = table.create(dimensionSizeArray[2], initialValue) end

	end
	
	return tensor
	
end

local function createIdentityTensor(dimensionSizeArray, locationArray)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local tensor = {}
	
	if (numberOfDimensions > 1) then
		
		for i = 1, dimensionSizeArray[1] do 
			
			local copiedLocationArray = table.clone(locationArray)
			
			table.insert(copiedLocationArray, i)
			
			local remainingDimensionSizeArray = {}

			for i = 2, #dimensionSizeArray do table.insert(remainingDimensionSizeArray, dimensionSizeArray[i]) end
			
			tensor[i] = createIdentityTensor(remainingDimensionSizeArray, copiedLocationArray) 
			
		end
		
	else
		
		for i = 1, dimensionSizeArray[1], 1 do
			
			local copiedLocationArray = table.clone(locationArray)
			
			local firstDimensionLocation = copiedLocationArray[1]
			
			tensor[i] = 1
			
			table.insert(copiedLocationArray, i)
			
			for _, location in ipairs(copiedLocationArray) do
				
				if (location ~= firstDimensionLocation) then
					
					tensor[i] = 0
					break
					
				end
				
			end
			
		end
		
	end
	
	return tensor
	
end

function AqwamTensorLibrary:createIdentityTensor(dimensionSizeArray)
	
	dimensionSizeArray = truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	return createIdentityTensor(dimensionSizeArray, {})
	
end

function AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(tensor)

	local dimensionSizeArray = {}

	for dimension = numberOfDimensions, 1, -1  do

		local length = getSubTensorLength(tensor, dimension)

		table.insert(dimensionSizeArray, length)

	end

	return dimensionSizeArray

end

function AqwamTensorLibrary:getNumberOfDimensions(tensor)
	
	if (typeof(tensor) ~= "table") then return 0 end

	return AqwamTensorLibrary:getNumberOfDimensions(tensor[1]) + 1
	
end

function AqwamTensorLibrary:add(...)

	local functionToApply = function(a, b) return (a + b) end

	return applyFunctionOnMultipleTensors(functionToApply, ...)

end

function AqwamTensorLibrary:subtract(...)

	local functionToApply = function(a, b) return (a - b) end

	return applyFunctionOnMultipleTensors(functionToApply, ...)

end

function AqwamTensorLibrary:multiply(...)

	local functionToApply = function(a, b) return (a * b) end

	return applyFunctionOnMultipleTensors(functionToApply, ...)

end

function AqwamTensorLibrary:divide(...)

	local functionToApply = function(a, b) return (a / b) end

	return applyFunctionOnMultipleTensors(functionToApply, ...)

end

function AqwamTensorLibrary:logarithm(...)

	return applyFunctionOnMultipleTensors(math.log, ...)

end

function AqwamTensorLibrary:exponent(...)

	return applyFunctionOnMultipleTensors(math.exp, ...)

end

function AqwamTensorLibrary:power(...)

	return applyFunctionOnMultipleTensors(math.pow, ...)

end

function AqwamTensorLibrary:printTensor(tensor)

	print("\n\n" .. generateTensorString(tensor) .. "\n\n")

end

function AqwamTensorLibrary:printTensorWithComma(tensor)

	print("\n\n" .. generateTensorStringWithComma(tensor) .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor(tensor)

	print("\n\n" .. generatePortableTensorString(tensor) .. "\n\n")

end

function AqwamTensorLibrary:copy(tensor)
	
	return deepCopyTable(tensor)
	
end

return AqwamTensorLibrary
