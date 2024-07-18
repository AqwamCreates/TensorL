local AqwamTensorLibrary = {}

local function removeFirstValueFromArray(array)
	
	local newArray = {}
	
	for i = 2, #array, 1 do table.insert(newArray, array[i]) end
	
	return newArray
	
end

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

local function get2DTensorTextSpacing(tensor, textSpacingArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray = get2DTensorTextSpacing(tensor[i], textSpacingArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray[i] = math.max(textSpacingArray[i], string.len(tostring(tensor[i]))) end

	end
	
	return textSpacingArray
	
end

function AqwamTensorLibrary:get2DTensorTextSpacing(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local sizeAtFinalDimension = dimensionSizeArray[numberOfDimensions]
	
	local textSpacingArray = table.create(sizeAtFinalDimension, 0)
	
	return get2DTensorTextSpacing(tensor, textSpacingArray)
	
end

function AqwamTensorLibrary:generateTensorString(tensor, textSpacingArray, dimensionDepth)

	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. AqwamTensorLibrary:generateTensorString(tensor[i], textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do
			
			local cellValue = tensor[i]
			
			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. " "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorStringWithComma(tensor, textSpacingArray, dimensionDepth)
	
	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then
		
		local spacing = ""

		text = text .. "{"
		
		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do
			
			if (i > 1) then text = text .. spacing end

			text = text .. AqwamTensorLibrary:generateTensorStringWithComma(tensor[i], textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText
		
			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generatePortableTensorString(tensor, textSpacingArray, dimensionDepth)

	dimensionDepth = dimensionDepth or 1

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. AqwamTensorLibrary:generatePortableTensorString(tensor[i], textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"
		
		if (dimensionDepth > 1) then text = text .. "," end

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " },"

	end

	return text

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


local function containAFalseBooleanInTensor(booleanTensor)

	local dimensionArray = AqwamTensorLibrary:getSize(booleanTensor)

	local numberOfValues = dimensionArray[1]

	local containsAFalseBoolean = true

	if (#dimensionArray > 1) then

		for i = 1, numberOfValues do containsAFalseBoolean = containAFalseBooleanInTensor(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 

			containsAFalseBoolean = (containsAFalseBoolean == booleanTensor[i])

			if (not containsAFalseBoolean) then return false end

		end

	end

	return containsAFalseBoolean

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

function AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)
	
	local expandedTensor = {}

	if (#dimensionSizeToAddArray > 1) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeToAddArray)
		
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

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = AqwamTensorLibrary:createTensor(remainingDimensionSizeArray, initialValue) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = table.create(dimensionSizeArray[2], initialValue) end

	end
	
	return tensor
	
end

function AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)
	
	mean = mean or 0

	standardDeviation = standardDeviation or 1
	
	local tensor = {}

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = AqwamTensorLibrary:createRandomNormalTensor(remainingDimensionSizeArray, mean, standardDeviation) end

	else

		for i = 1, dimensionSizeArray[1], 1 do 
			
			local randomNumber1 = math.random()

			local randomNumber2 = math.random()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.
			
			tensor[i] = (zScore * standardDeviation) + mean
			
		end

	end

	return tensor

end

function AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

	local tensor = {}

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = AqwamTensorLibrary:createRandomNormalTensor(remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random() end

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
			
			local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)
			
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

function AqwamTensorLibrary:getTotalSize(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local totalSize = 1
	
	for _, value in ipairs(dimensionSizeArray) do
		
		totalSize = value * totalSize
		
	end
	
	return totalSize
	
end

function AqwamTensorLibrary:dotProduct(tensor1, tensor2) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc
	
	
	
end

function AqwamTensorLibrary:mean(tensor, dimension)
	
	if (not dimension) then
		
		local sum = fullSum(tensor)
		
		local totalSize = AqwamTensorLibrary:getTotalSize(tensor)
		
		local mean = sum / totalSize
		
		return mean
		
	end
	
	
end
	
function AqwamTensorLibrary:standardDeviation(tensor, dimension)
	
	
end

function AqwamTensorLibrary:zScoreNormalize(tensor, dimension)
	
	
end

function AqwamTensorLibrary:findMaximumValue(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local value

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do value = AqwamTensorLibrary:findMaximumValue(tensor[i]) end

	else

		value = math.max(table.unpack(tensor))

	end

	return value
	
end

function AqwamTensorLibrary:findMinimumValue(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local value

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do value = AqwamTensorLibrary:findMinimumValue(tensor[i]) end

	else

		value = math.min(table.unpack(tensor))

	end

	return value
	
end

local function flatten(tensor, targetTensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do flatten(tensor[i], targetTensor) end

	else
		
		for _, value in ipairs(tensor) do table.insert(targetTensor, value) end

	end

	return tensor
	
end

function AqwamTensorLibrary:flatten(tensor)
	
	local flattenedTensor = {}
	
	flatten(tensor, flattenedTensor)
	
	return flattenedTensor
	
end

local function reshape(flattenedTensor, dimensionSizeArray, dimensionIndex)
	
	local tensor = {}
	
	dimensionIndex = dimensionIndex or 0

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 
			
			tensor[i], dimensionIndex = reshape(flattenedTensor, remainingDimensionSizeArray, dimensionIndex) 
			
		end

	else
		
		for i = 1, dimensionSizeArray[1], 1 do 
			
			dimensionIndex = dimensionIndex + 1
			table.insert(tensor, flattenedTensor[dimensionIndex]) 
			
		end

	end
	
	return tensor, dimensionIndex
	
end

function AqwamTensorLibrary:reshape(flattenedTensor, dimensionSizeArray)
	
	local flattenedTensorSizeArray = AqwamTensorLibrary:getSize(flattenedTensor)

	if (#flattenedTensorSizeArray > 1) then error("Unable to reshape a tensor that has more than one dimension.") end
	
	local totalNumberOfValuesRequired = 1
	
	for _, value in ipairs(dimensionSizeArray) do
		
		totalNumberOfValuesRequired = totalNumberOfValuesRequired * value
		
	end
	
	if (totalNumberOfValuesRequired ~= flattenedTensorSizeArray[1]) then error("The number of values in flattened tensor does not equal to total number of values of the reshaped tensor.") end
	
	local tensor = reshape(flattenedTensor, dimensionSizeArray)
	
	return tensor
	
end

local function getOutOfBoundsIndexArray(array, arrayToBeCheckedForOutOfBounds)

	local outOfBoundsIndexArray = {}

	for i, value in ipairs(arrayToBeCheckedForOutOfBounds) do

		if (value < 1) or (value > array[i]) then table.insert(outOfBoundsIndexArray, i) end

	end

	return outOfBoundsIndexArray

end

local function getFalseBooleanIndexArray(functionToApply, array1, array2)

	local falseBooleanIndexArray = {}

	for i, value in ipairs(array1) do

		if (not functionToApply(value, array2[i])) then table.insert(falseBooleanIndexArray, i) end

	end

	return falseBooleanIndexArray

end

local function extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local remainingOriginDimensionIndexArray = removeFirstValueFromArray(originDimensionIndexArray)

	local remainingTargetDimensionIndexArray = removeFirstValueFromArray(targetDimensionIndexArray)
	
	local extractedTensor = {}
	
	if (#dimensionSizeArray > 1) then

		for i = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do 

			extractedTensor[i] = extract(tensor[i], remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray)

		end

	else
		
		for i = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do 

			table.insert(extractedTensor, tensor[i]) 

		end

	end
	
	return extractedTensor
	
end

function AqwamTensorLibrary:extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #originDimensionIndexArray) then error("Invalid origin dimension index array.") end

	if (numberOfDimensions ~= #targetDimensionIndexArray) then error("Invalid target dimension index array.") end

	local outOfBoundsOriginIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, originDimensionIndexArray)

	local outOfBoundsTargetIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, targetDimensionIndexArray)

	local falseBooleanIndexArray = getFalseBooleanIndexArray(function(a, b) return (a <= b) end, originDimensionIndexArray, targetDimensionIndexArray)

	local outOfBoundsOriginIndexArraySize = #outOfBoundsOriginIndexArray

	local outOfBoundsTargetIndexArraySize = #outOfBoundsTargetIndexArray

	local falseBooleanIndexArraySize = #falseBooleanIndexArray

	if (outOfBoundsOriginIndexArraySize > 0) then

		local errorString = "Attempting to set an origin dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsOriginIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsOriginIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end
	
	return extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
end

function AqwamTensorLibrary:concatenate(tensor1, tensor2, dimension)
	
	
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

function AqwamTensorLibrary:isSameTensor(tensor1, tensor2)

	local booleanTensor = AqwamTensorLibrary:isEqualTo(tensor1, tensor2)
	
	return containAFalseBooleanInTensor(booleanTensor)

end

function AqwamTensorLibrary:isEqualTo(tensor1, tensor2)

	local functionToApply = function(a, b) return (a == b) end

	local result = applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary:isGreaterThan(tensor1, tensor2)

	local functionToApply = function(a, b) return (a > b) end

	return applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(tensor1, tensor2)

	local functionToApply = function(a, b) return (a >= b) end

	return applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

end

function AqwamTensorLibrary:isLessThan(tensor1, tensor2)

	local functionToApply = function(a, b) return (a < b) end

	return applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

end

function AqwamTensorLibrary:isLessOrEqualTo(tensor1, tensor2)

	local functionToApply = function(a, b) return (a <= b) end

	return applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

end

function AqwamTensorLibrary:printTensor(tensor)
	
	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	print("\n\n" .. AqwamTensorLibrary:generateTensorString(tensor, textSpacingArray) .. "\n\n")

end

function AqwamTensorLibrary:printTensorWithComma(tensor)
	
	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	print("\n\n" .. AqwamTensorLibrary:generateTensorStringWithComma(tensor, textSpacingArray) .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor(tensor)
	
	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	print("\n\n" .. AqwamTensorLibrary:generatePortableTensorString(tensor, textSpacingArray) .. "\n\n")

end

function AqwamTensorLibrary:copy(tensor)
	
	return deepCopyTable(tensor)
	
end

return AqwamTensorLibrary
