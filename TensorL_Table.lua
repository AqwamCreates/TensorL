--[[

	--------------------------------------------------------------------

	Version 1.0.0

	Aqwam's Tensor Library (TensorL)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our terms and conditions at:
	
	https://github.com/AqwamCreates/TensorL/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT WITHOUT PERMISSION!
	
	--------------------------------------------------------------------

--]]

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

function AqwamTensorLibrary:truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	while true do

		local size = newDimensionSizeArray[1]

		if (size ~= 1) then break end

		table.remove(newDimensionSizeArray, 1)

	end

	return newDimensionSizeArray

end

function AqwamTensorLibrary:truncateTensorIfRequired(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (dimensionSizeArray[1] == 1) then

		return AqwamTensorLibrary:truncateTensorIfRequired(tensor[1])

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
	
	local firstTensor = AqwamTensorLibrary:truncateTensorIfRequired(firstTensor)

	if (numberOfTensors == 1) then 
		
		local tensor = applyFunctionUsingOneTensor(functionToApply, firstTensor)
		
		tensor = AqwamTensorLibrary:truncateTensorIfRequired(tensor)
		
		return tensor
		
	end

	local tensor = firstTensor

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]
		
		local otherTensor = AqwamTensorLibrary:truncateTensorIfRequired(otherTensor)

		tensor, otherTensor = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor, otherTensor)

		tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor)

	end
	
	tensor = AqwamTensorLibrary:truncateTensorIfRequired(tensor)

	return tensor

end

local function getTensorDimensionSizeArrayString(dimensionSizeArray)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local tensorDimensionSizeArrayString = "("
	
	for s, size in ipairs(dimensionSizeArray) do

		tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. size

		if (s ~= numberOfDimensions) then

			tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ", "

		end

	end
	
	tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ")"
	
	return tensorDimensionSizeArrayString
	
end

local function onBroadcastError(dimensionSizeArray1, dimensionSizeArray2)
	
	local tensor1DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray1)
	
	local tensor2DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray2)

	local errorMessage = "Unable to broadcast. \n" .. "Tensor 1 size: " .. tensor1DimensionSizeArrayString .."\n" .. "Tensor 2 size: " .. tensor2DimensionSizeArrayString .."\n"
	
	error(errorMessage)

end

function AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)
	
	local expandedTensor = {}

	if (#dimensionSizeToAddArray > 1) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeToAddArray)
		
		for i = 1, dimensionSizeToAddArray[1], 1 do expandedTensor[i] = AqwamTensorLibrary:expand(tensor, remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeToAddArray[1], 1 do expandedTensor[i] = deepCopyTable(tensor) end

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

local function createTensor(dimensionSizeArray, initialValue)
	
	local tensor = {}

	if (#dimensionSizeArray >= 3) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createTensor(remainingDimensionSizeArray, initialValue) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = table.create(dimensionSizeArray[2], initialValue) end

	end
	
	return tensor
	
end

function AqwamTensorLibrary:createTensor(dimensionSizeArray, initialValue) -- Do not truncate the dimension size array! It is needed by the hardcoded transpose operation!
	
	initialValue = initialValue or 0
	
	return createTensor(dimensionSizeArray, initialValue)
	
end

local function createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)
	
	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomNormalTensor(remainingDimensionSizeArray, mean, standardDeviation) end

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

function AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)
	
	dimensionSizeArray = AqwamTensorLibrary:truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	mean = mean or 0

	standardDeviation = standardDeviation or 1
	
	return createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

end

local function createRandomUniformTensor(dimensionSizeArray)
	
	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomNormalTensor(remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random() end

	end

	return tensor
	
end

function AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
	
	dimensionSizeArray = AqwamTensorLibrary:truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	return createRandomUniformTensor(dimensionSizeArray)

end

local function createIdentityTensor(dimensionSizeArray, locationArray)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local tensor = {}
	
	if (numberOfDimensions >= 2) then
		
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
	
	dimensionSizeArray = AqwamTensorLibrary:truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
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

--[[

local function transpose(originalTensor, tensor, targetTensor, dimensionIndexArray, currentDimension)

	currentDimension = currentDimension or 0

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (currentDimension ~= dimensionIndexArray[1]) then
		
		for i = 1, dimensionSizeArray[1], 1 do 
			
			table.insert(targetTensor, transpose(originalTensor, tensor[i], targetTensor, dimensionIndexArray, currentDimension + 1))
			
		end

	else
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = goToSubTensor(originalTensor, dimensionIndexArray[2]) end

		--for i = 1, dimensionSizeArray[1], 1 do   end

	end

	return targetTensor

end

--]]

--[[

local function transpose(originalTensor, dimensionIndexArray, currentDimension)
	
	currentDimension = currentDimension or 1
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(originalTensor)
	
	print(currentDimension == dimensionIndexArray[1])
	
	print(currentDimension)
	
	if (currentDimension == dimensionIndexArray[1]) then
		
		for i = 1, #dimensionSizeArray, 1 do originalTensor[i] = AqwamTensorLibrary:extract(originalTensor[i],  dimensionIndexArray) end
		
	else 
		
		for i = 1, #dimensionSizeArray, 1 do originalTensor[i] = transpose(originalTensor[i],  dimensionIndexArray, currentDimension + 1) end
		
	end
	
	return originalTensor
	
end

--]]

--[[

local function transpose(tensor, targetTensor, originDimensionIndex, targetDimensionIndex, currentDimension)
	
	currentDimension = currentDimension or 0
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if (originDimensionIndex ~= currentDimension) then
		
		print(dimensionSizeArray[1])
		
		print(targetTensor)
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = goToSubTensor(tensor[i], targetDimensionIndex) end
		
	else
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = transpose(tensor[i], targetTensor[i], originDimensionIndex, targetDimensionIndex, currentDimension + 1) end
		
	end
	
	return targetTensor
	
end

--]]


local function hardcodedTranspose(tensor, dimensionIndexArray) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionArray
	
	local offset = 5 - numberOfDimensions
	
	local dimensionSizeToAddArray = table.create(offset, 1)
	
	local expandedTensor = AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)
	
	local dimension1 = dimensionIndexArray[1] + offset
	local dimension2 = dimensionIndexArray[2] + offset
	
	local newDimensionIndexArray = {dimension1, dimension2}
	
	local expandedDimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor)

	expandedDimensionSizeArray[dimension1], expandedDimensionSizeArray[dimension2] = expandedDimensionSizeArray[dimension2], expandedDimensionSizeArray[dimension1]
	
	local newTensor = AqwamTensorLibrary:createTensor(expandedDimensionSizeArray, true)

	if (table.find(newDimensionIndexArray, 1)) and (table.find(newDimensionIndexArray, 2)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[b][a][c][d][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 1)) and (table.find(newDimensionIndexArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[c][b][a][d][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 2)) and (table.find(newDimensionIndexArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do
							
							newTensor[a][b][c][d][e] = expandedTensor[a][c][b][d][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 1)) and (table.find(newDimensionIndexArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[d][b][c][a][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 1)) and (table.find(newDimensionIndexArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[e][b][c][d][a]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 2)) and (table.find(newDimensionIndexArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][d][c][b][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 2)) and (table.find(newDimensionIndexArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][e][c][d][b]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 3)) and (table.find(newDimensionIndexArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][d][c][e]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 3)) and (table.find(newDimensionIndexArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][e][d][c]
							
						end

					end

				end

			end

		end

	elseif (table.find(newDimensionIndexArray, 4)) and (table.find(newDimensionIndexArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][c][e][d]
							
						end

					end

				end

			end

		end

	else
		
		error("Invalid dimensions!")
		
	end
	
	return AqwamTensorLibrary:truncateTensorIfRequired(newTensor)
	
end

function AqwamTensorLibrary:transpose(tensor, dimensionIndexArray)
	
	if (#dimensionIndexArray ~= 2) then error("Dimension index array must contain exactly 2 dimensions.") end
	
	if (dimensionIndexArray[1] == dimensionIndexArray[2]) then return tensor end
	
	return hardcodedTranspose(tensor, dimensionIndexArray)
	
end

local function dotProduct(tensor1, tensor2)

	local tensor1DimensionSizeArray = AqwamTensorLibrary:getSize(tensor1)

	local tensor2DimensionSizeArray = AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray

	local tensor = {}

	if numberOfDimensions1 >= 2 and numberOfDimensions2 >= 2 then

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i]) end

	elseif numberOfDimensions1 == 1 and numberOfDimensions2 >= 2 then

		for i = 1, tensor2DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1, tensor2[i]) end

	else

		local sum = 0

		if numberOfDimensions1 == 1 and numberOfDimensions2 == 1 then

			for i = 1, #tensor1 do sum = sum + tensor1[i] * tensor2[i] end

			tensor = sum

		elseif numberOfDimensions1 == 1 and numberOfDimensions2 == 2 then

			local tensor2Column = #tensor2[1]

			for column = 1, tensor2Column do

				local columnSum = 0

				for i = 1, #tensor1 do columnSum = columnSum + tensor1[i] * tensor2[i][column] end

				tensor[column] = columnSum

			end

		end

	end

	return tensor

end

local function convertTensorToScalar(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if (#dimensionSizeArray >= 1) then
		
		return convertTensorToScalar(tensor[1])
		
	else
		
		return tensor[1]
		
	end
	
end

function AqwamTensorLibrary:dotProduct(...) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc
	
	local tensorArray = {...}
	
	local tensor = tensorArray[1]
	
	for i = 2, #tensorArray, 1 do
		
		local otherTensor = tensorArray[i]
		
		local tensorDimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
		
		local otherTensorDimensionSizeArray = AqwamTensorLibrary:getSize(otherTensor)
		
		local tensorNumberOfDimensions = #tensorDimensionSizeArray
		
		local otherTensorNumberOfDimensions = #otherTensorDimensionSizeArray
		
		if (tensorNumberOfDimensions == otherTensorNumberOfDimensions) and (tensorNumberOfDimensions ~= 1) and (otherTensorNumberOfDimensions ~= 1) then
			
			if (tensorDimensionSizeArray[tensorNumberOfDimensions] ~= otherTensorDimensionSizeArray[otherTensorNumberOfDimensions - 1]) then error("The size of the last dimension of tensor " .. (i - 1) .. " is not equal to the size of second last dimension of the tensor " .. i .. ".") end

			for j = 1, (otherTensorNumberOfDimensions - 2), 1 do

				if (tensorDimensionSizeArray[i] ~= otherTensorDimensionSizeArray[i]) then error("The size of dimension " .. j .. " of tensor " .. (i - 1) .. " is not equal to the size of dimension " .. j .. " of the tensor " .. i .. ".") end

			end
			
		elseif (tensorNumberOfDimensions == 1) and (otherTensorNumberOfDimensions >= 2) then
			
			for j = (otherTensorNumberOfDimensions - 1), otherTensorNumberOfDimensions, 1 do

				if (tensorDimensionSizeArray[1] ~= otherTensorDimensionSizeArray[j]) then error("The size of dimension 1 of tensor " .. (i - 1) .. " is not equal to the size of dimension " .. j .. " of the tensor " .. i .. ".") end

			end
			
		elseif (tensorNumberOfDimensions == 1) and (otherTensorNumberOfDimensions == 1) then
			
			if (tensorDimensionSizeArray[1] ~= otherTensorDimensionSizeArray[1]) then error("The size of dimension 1 of tensor " .. (i - 1) .. " is not equal to the size of dimension 1 of the tensor " .. i .. ".") end
			
		end
		
		tensor = dotProduct(tensor, otherTensor)
		
	end
	
	local resultTensorDimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if (resultTensorDimensionSizeArray == nil) then return tensor end -- If our tensor is actually a scalar, just return the number.
	
	for _, size in ipairs(resultTensorDimensionSizeArray) do -- Return the original tensor if any dimension sizes are not equal to 1.
		
		if (size ~= 1) then return AqwamTensorLibrary:truncateTensorIfRequired(tensor) end
		
	end
	
	return convertTensorToScalar(tensor)
	
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

--[[

local function dimensionSum(tensor, dimension)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local newTensor = {}

	if (#dimensionSizeArray ~= dimension) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = dimensionSum(tensor[i], dimension) end

	else

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = 0 end

		-- Sum along the specified dimension
		for i = 1, #tensor do
			
			for j = 1, dimensionSizeArray[1], 1 do
				
				print(tensor)
				
				newTensor[j] = newTensor[j] + tensor[j]
				
			end
			
		end

	end

	return newTensor

end

--]]

--[[

local function dimensionSum(tensor, dimension)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local firstDimensionSize = dimensionSizeArray[1]
	
	local secondDimensionSize = dimensionSizeArray[2]
	
	local newTensor = table.create(firstDimensionSize, 0)
	
	if (secondDimensionSize ~= dimension) and (numberOfDimensions >= 2) then
		
		for i = 1, firstDimensionSize, 1 do newTensor[i] = dimensionSum(tensor[i], dimension) end
		
	--else
		
		--for i = 1, firstDimensionSize, 1 do newTensor[i] = newTensor[i] + tensor[i] end
		
	end

	return newTensor

end

--]]

--[[

local function dimensionSum(tensor, dimension, locationArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = table.create(dimensionSizeArray[1], 0)

	if (numberOfDimensions ~= dimension) then

		for i = 1, dimensionSizeArray[1] do 

			local copiedLocationArray = table.clone(locationArray)

			table.insert(copiedLocationArray, i)

			newTensor[i] = dimensionSum(tensor[i], dimension, copiedLocationArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do
			
			newTensor[i] = newTensor[i] + tensor[i]

		end

	end

	return newTensor

end

--]]

--[[

local function dimensionSum(tensor, targetDimension, currentDimension)
	
	currentDimension = currentDimension or 1
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray

	--local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}
	
	print(currentDimension == targetDimension)
	
	print(numberOfDimensions == 0)
	
	if (currentDimension == targetDimension) then
		
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = dimensionSum(tensor[i], targetDimension, currentDimension + 1) end
		
	elseif (numberOfDimensions == 0) then
			
		return tensor
		
	else
		
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = tensor[i] end
		
	end
	
	return newTensor
	
end

--]]

local function hardcodedDimensionSum(tensor, dimension) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!
	
	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionArray

	local offset = 5 - numberOfDimensions
	
	local dimension = dimension + offset

	local dimensionSizeToAddArray = table.create(offset, 1)

	local expandedTensor = AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)

	local expandedDimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor)
	
	local expandedSumDimensionArray = table.clone(expandedDimensionSizeArray)
	
	expandedSumDimensionArray[dimension] = 1

	local newTensor = AqwamTensorLibrary:createTensor(expandedSumDimensionArray, 0)
	
	if (dimension == 1) then
		
		for a = 1, expandedDimensionSizeArray[1], 1 do
			
			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[1][b][c][d][e] = newTensor[1][b][c][d][e] + expandedTensor[a][b][c][d][e]
							
						end

					end

				end

			end
			
		end
		
	elseif (dimension == 2) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do
							
							print()

							newTensor[a][1][c][d][e] = newTensor[a][1][c][d][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end
		
	elseif (dimension == 3) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][1][d][e] = newTensor[a][b][1][d][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end
		
	elseif (dimension == 4) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][1][e] = newTensor[a][b][c][1][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end
		
	elseif (dimension == 5) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][1] = newTensor[a][b][c][d][1] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end
		
	end
	
	return AqwamTensorLibrary:truncateTensorIfRequired(newTensor)
	
end

function AqwamTensorLibrary:sum(tensor, dimension)

	if (not dimension) then return fullSum(tensor) end

	local numberOfDimensions = #AqwamTensorLibrary:getSize(tensor)

	if (dimension <= 0) or (dimension > numberOfDimensions) then error("Invalid dimensions.") end

	return hardcodedDimensionSum(tensor, dimension)

end

function AqwamTensorLibrary:mean(tensor, dimension)
	
	local sumTensor = AqwamTensorLibrary:sum(tensor, dimension)
	
	
end
	
function AqwamTensorLibrary:standardDeviation(tensor, dimension)
	
	
end

function AqwamTensorLibrary:zScoreNormalize(tensor, dimension)
	
	
end

function AqwamTensorLibrary:findMaximumValue(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local value

	if (#dimensionSizeArray >= 2) then

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

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do value = AqwamTensorLibrary:findMinimumValue(tensor[i]) end

	else

		value = math.min(table.unpack(tensor))

	end

	return value
	
end

local function flatten(tensor, targetTensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (#dimensionSizeArray >= 2) then

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

	if (#dimensionSizeArray >= 2) then

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
	
	if (#dimensionSizeArray >= 2) then

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

	if (outOfBoundsTargetIndexArraySize > 0) then

		local errorString = "Attempting to set an target dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsTargetIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsTargetIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	if (falseBooleanIndexArraySize > 0) then

		local errorString = "The origin dimension index is larger than the target dimension index for dimensions at "

		for i, index in ipairs(outOfBoundsOriginIndexArray) do

			errorString = errorString .. index

			if (i < falseBooleanIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end
	
	local extractedTensor = extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
	return AqwamTensorLibrary:truncateTensorIfRequired(extractedTensor)
	
end

local function concatenate(targetTensor, otherTensor, targetDimension, currentDimension)
	
	currentDimension = currentDimension or 1

	if (currentDimension ~= targetDimension) then
		
		local dimensionSizeArray = AqwamTensorLibrary:getSize(targetTensor)
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = concatenate(targetTensor[i], otherTensor[i], targetDimension, currentDimension + 1) end

	else
		
		for _, value in ipairs(otherTensor) do table.insert(targetTensor, value) end

	end
	
	return targetTensor
	
end

function AqwamTensorLibrary:concatenate(tensor1, tensor2, dimension)
	
	local tensor1 = AqwamTensorLibrary:truncateTensorIfRequired(tensor1)
	
	local tensor2 = AqwamTensorLibrary:truncateTensorIfRequired(tensor2)
	
	local dimensionSizeArray1 = AqwamTensorLibrary:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary:getSize(tensor2)
	
	local numberOfDimensions1 = #dimensionSizeArray1
	
	local numberOfDimensions2 = #dimensionSizeArray2
	
	if (numberOfDimensions1 ~= numberOfDimensions2) then error("The tensors do not have equal number of dimensions.") end
	
	if (numberOfDimensions1 <= 0) or (dimension > numberOfDimensions1) then error("The selected dimension is out of bounds.") end

	for dimensionIndex = 1, numberOfDimensions1, 1 do

		if (dimensionIndex == dimension) then continue end

		if (dimensionSizeArray1[dimensionIndex] ~= dimensionSizeArray2[dimensionIndex]) then error("The tensors do not contain equal dimension values at dimension " .. dimensionIndex .. ".") end

	end
	
	local targetTensor = deepCopyTable(tensor1)
	
	return concatenate(targetTensor, tensor2, dimension)
	
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

	return applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

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

local function applyFunction(functionToApply, ...)
	
	local tensorArray = {...}
	
	local newTensor = {}
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensorArray[1])
	
	if (#dimensionSizeArray >= 2) then

		for i = 1, dimensionSizeArray[1], 1 do 
			
			local subTensorArray = {}
			
			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end
			
			newTensor[i] = applyFunction(functionToApply, table.unpack(subTensorArray)) 
			
		end

	else
		
		for i = 1, dimensionSizeArray[1], 1 do 
			
			local subTensorArray = {}

			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end
			
			newTensor[i] = functionToApply(table.unpack(subTensorArray)) 
			
		end

	end
	
	return newTensor
	
end

function AqwamTensorLibrary:applyFunction(functionToApply, ...)
	
	local tensorArray = {...}
	
	local allDimensionSizeArrays = {}
	
	for _, tensor in ipairs(tensorArray) do
		
		local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
		
		table.insert(allDimensionSizeArrays, dimensionSizeArray)
		
	end
	
	local firstDimensionSizeArray = allDimensionSizeArrays[1]
	
	for i = 2, #tensorArray, 1 do
		
		local dimensionSizeArray = allDimensionSizeArrays[i]
		
		if (#firstDimensionSizeArray ~= #dimensionSizeArray) then error("Tensor ".. (i - 1) .. " and " .. i .. " does not have the same number of dimensions.") end
		
		for s, size in ipairs(firstDimensionSizeArray) do
			
			if (size ~= dimensionSizeArray[s]) then error("Tensor " .. (i - 1) .. " and " .. i .. " does not contain equal dimension values at dimension " .. s .. ".") end
			
		end
		
	end
	
	return applyFunction(functionToApply, ...)
	
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
